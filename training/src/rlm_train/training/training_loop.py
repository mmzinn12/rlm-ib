"""Readable attempt → feedback → selection → scoring → loss → update loop."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from rlm_train.attempts import AnnotatedAttempt, AttemptRunner
from rlm_train.datasets.records import DatasetRecord
from rlm_train.feedback import FeedbackCollector
from rlm_train.feedback.feedback_records import FeedbackBundle
from rlm_train.generation.generated_text import GeneratedText
from rlm_train.metrics.schema import MetricObservation
from rlm_train.student import TrainableStudent
from rlm_train.token_selection import (
    TokenSelection,
    TokenSelectionResult,
    choose_tokens_many,
    selection_for_schema_v1,
)
from rlm_train.training.prepare_batch import (
    LossResult,
    StudentPredictionBatch,
    TrainingBatch,
)
from rlm_train.training.requirements import TrainingRequirements, combine_requirements
from rlm_train.training.training_state import TrainingState
from rlm_train.trajectory.schema import FeedbackRecord


class DatasetSource(Protocol):
    def records(self) -> Sequence[DatasetRecord]: ...


class TrainingMethod(Protocol):
    name: str
    weight: float
    requirements: TrainingRequirements

    def calculate_loss(self, batch: TrainingBatch) -> LossResult: ...


class CheckpointWriter(Protocol):
    def write(self, state: TrainingState, *, final: bool) -> object | None: ...


class AttemptWriter(Protocol):
    def write(self, attempt: AnnotatedAttempt) -> object | None: ...


@dataclass(frozen=True)
class TrainingResult:
    state: TrainingState
    final_loss: float
    attempt_artifacts: tuple[str, ...] = ()
    checkpoint_artifacts: tuple[str, ...] = ()


class TrainingLoop:
    """Keep the conceptual training operations explicit and in execution order."""

    def __init__(
        self,
        *,
        run_settings: Any,
        dataset: DatasetSource,
        attempt_runner: AttemptRunner,
        feedback_collector: FeedbackCollector,
        uncertainty_provider: Any | None = None,
        student: TrainableStudent,
        training_methods: Iterable[TrainingMethod],
        optimizer: Any,
        scheduler: Any | None = None,
        attempt_writer: AttemptWriter | None = None,
        metric_sink: Any | None = None,
        checkpoint_writer: CheckpointWriter | None = None,
        initial_state: TrainingState | None = None,
        precision_context: Callable[[], AbstractContextManager[Any]] | None = None,
        gradient_scaler: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self.run_settings = run_settings
        self.dataset = dataset
        self.attempt_runner = attempt_runner
        self.feedback_collector = feedback_collector
        self.uncertainty_provider = uncertainty_provider
        self.student = student
        self.training_methods = tuple(training_methods)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.attempt_writer = attempt_writer
        self.metric_sink = metric_sink
        self.checkpoint_writer = checkpoint_writer
        self.initial_state = initial_state
        self.precision_context = precision_context or nullcontext
        self.gradient_scaler = gradient_scaler
        self.verbose = verbose
        if not self.training_methods:
            raise ValueError("training loop requires at least one enabled training method")
        if self.run_settings.uncertainty.enabled and self.uncertainty_provider is None:
            raise ValueError("enabled uncertainty measurement requires an uncertainty provider")

    def train(self) -> TrainingResult:
        from rlm_train.sdpo import score_with_feedback

        method_requirements = {method.name: method.requirements for method in self.training_methods}
        requirements = combine_requirements(method_requirements)
        records = tuple(self.dataset.records())
        if not records:
            raise ValueError("training dataset is empty")
        runtime = self.run_settings.runtime
        state = self.initial_state or TrainingState(
            run_spec_fingerprint=self.run_settings.fingerprint
        )
        if state.run_spec_fingerprint != self.run_settings.fingerprint:
            raise ValueError("initial training state does not match the run settings")

        record_index = state.examples_seen
        final_loss = 0.0
        attempt_paths: list[str] = []
        checkpoint_paths: list[str] = []
        parameters = tuple(self.student.trainable_parameters())

        for optimizer_step in range(state.optimizer_step + 1, runtime.max_optimizer_steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(runtime.gradient_accumulation_steps):
                record, attempts, selections = self.choose_trainable_example(
                    records,
                    record_index,
                    requirements.attempt_count,
                )
                record_index += 1

                feedback = self.feedback_collector.collect(attempts, requirements.feedback_scopes)
                if self.run_settings.uncertainty.enabled:
                    measurements = tuple(
                        measurement
                        for attempt in attempts
                        for measurement in self.uncertainty_provider.assess_rollout(record, attempt)
                    )
                    feedback = feedback.model_copy(update={"uncertainty_assessments": measurements})
                    self.record_uncertainty_metrics(optimizer_step, measurements)
                student_predictions = {
                    method.name: score_selected_tokens(
                        student=self.student,
                        attempts=attempts,
                        selections={
                            attempt_id: result.selection
                            for attempt_id, result in selections[method.name].items()
                        },
                    )
                    for method in self.training_methods
                }
                feedback_predictions = {
                    method.name: (
                        score_with_feedback(
                            student=self.student,
                            attempts=attempts,
                            feedback=feedback,
                            selections={
                                attempt_id: result.selection
                                for attempt_id, result in selections[method.name].items()
                            },
                            included_text=method.requirements.included_text,
                            top_k=int(getattr(method, "top_k", 32)),
                        )
                        if method.requirements.feedback_predictions
                        else {}
                    )
                    for method in self.training_methods
                }
                with self.precision_context():
                    losses = {
                        method.name: method.calculate_loss(
                            TrainingBatch(
                                attempts=attempts,
                                token_selections={
                                    attempt_id: result.selection
                                    for attempt_id, result in selections[method.name].items()
                                },
                                student_predictions=student_predictions[method.name],
                                feedback=feedback,
                                feedback_predictions=feedback_predictions[method.name],
                            )
                        )
                        for method in self.training_methods
                    }
                    total_loss = sum(
                        (
                            losses[method.name].loss * method.weight
                            for method in self.training_methods
                        ),
                        start=0,
                    )
                    scaled_loss = total_loss / runtime.gradient_accumulation_steps
                loss_value = finite_scalar(total_loss, "combined loss")
                self.backward(scaled_loss)
                accumulated_loss += loss_value / runtime.gradient_accumulation_steps
                attempt_paths.extend(
                    self.write_attempts(
                        attempts,
                        selections=selections,
                        feedback=feedback,
                        feedback_predictions=feedback_predictions,
                        losses=losses,
                    )
                )
                self.verbose_print(
                    f"record={record.record_id} attempts={len(attempts)} loss={loss_value:.6f}"
                )

            gradient_norm = update_student(
                optimizer=self.optimizer,
                parameters=parameters,
                max_gradient_norm=runtime.max_gradient_norm,
                gradient_scaler=self.gradient_scaler,
            )
            if self.scheduler is not None:
                self.scheduler.step()
            final_loss = accumulated_loss
            state = TrainingState(
                optimizer_step=optimizer_step,
                examples_seen=record_index,
                run_spec_fingerprint=self.run_settings.fingerprint,
            )
            self.record_metrics(optimizer_step, final_loss, gradient_norm)
            if self.checkpoint_writer is not None:
                path = self.checkpoint_writer.write(
                    state,
                    final=optimizer_step == runtime.max_optimizer_steps,
                )
                if path is not None:
                    checkpoint_paths.append(str(path))

        return TrainingResult(
            state=state,
            final_loss=final_loss,
            attempt_artifacts=tuple(attempt_paths),
            checkpoint_artifacts=tuple(checkpoint_paths),
        )

    def choose_trainable_example(
        self,
        records: tuple[DatasetRecord, ...],
        record_index: int,
        attempt_count: int,
    ) -> tuple[
        DatasetRecord,
        tuple[AnnotatedAttempt, ...],
        dict[str, dict[str, TokenSelectionResult]],
    ]:
        for offset in range(len(records)):
            record = records[(record_index + offset) % len(records)]
            attempts = self.attempt_runner.run_many(record, count=attempt_count)
            for attempt in attempts:
                if attempt.policy.get("policy_owner") != self.student.model_info.student_id:
                    raise ValueError("attempt was not generated by the configured student")
            selections = {
                method.name: choose_tokens_many(
                    attempts,
                    training_method=method.name,
                    included_text=method.requirements.included_text,
                    student_id=self.student.model_info.student_id,
                )
                for method in self.training_methods
            }
            if all(
                result.selection.active_token_count > 0
                for method_selections in selections.values()
                for result in method_selections.values()
            ):
                return record, attempts, selections
        raise ValueError(
            "no dataset record produced trainable tokens; check included_text and whether "
            "the student asks addressable helper questions"
        )

    def backward(self, loss: Any) -> None:
        finite_scalar(loss, "scaled loss")
        if self.gradient_scaler is not None:
            self.gradient_scaler.scale(loss).backward()
        else:
            loss.backward()

    def write_attempts(
        self,
        attempts: tuple[AnnotatedAttempt, ...],
        *,
        selections: dict[str, dict[str, TokenSelectionResult]],
        feedback: FeedbackBundle,
        feedback_predictions: dict[str, dict[str, Any]],
        losses: dict[str, LossResult],
    ) -> list[str]:
        if self.attempt_writer is None:
            return []
        paths = []
        for attempt in attempts:
            durable_selections = {
                method_name: selection_for_schema_v1(
                    method_selections[attempt.rollout_id].selection,
                    attempt,
                    included_text=next(
                        method.requirements.included_text
                        for method in self.training_methods
                        if method.name == method_name
                    ),
                    student_id=self.student.model_info.student_id,
                )
                for method_name, method_selections in selections.items()
            }
            annotations = attempt.annotations.model_copy(
                update={"objective_selections": durable_selections}
            )
            stored_predictions = tuple(
                values[attempt.rollout_id].model_dump(mode="json")
                for values in feedback_predictions.values()
                if attempt.rollout_id in values
            )
            method_records = {
                name: {
                    "active_token_count": result.active_token_count,
                    "diagnostics": result.diagnostics,
                }
                for name, result in losses.items()
            }
            enriched = attempt.model_copy(
                update={
                    "annotations": annotations,
                    "feedback": feedback_record(feedback),
                    # Schema v1 retains this persisted field name for compatibility.
                    "teacher_targets": stored_predictions,
                    "objectives": method_records,
                }
            )
            path = self.attempt_writer.write(enriched)
            if path is not None:
                paths.append(str(path))
        return paths

    def record_metrics(self, step: int, loss: float, gradient_norm: float) -> None:
        if self.metric_sink is None:
            return
        for observation in (
            MetricObservation(name="train/loss/total", value=loss, step=step),
            MetricObservation(name="train/optimizer/gradient_norm", value=gradient_norm, step=step),
        ):
            self.metric_sink.write(observation)

    def record_uncertainty_metrics(self, step: int, measurements: tuple[Any, ...]) -> None:
        if self.metric_sink is None:
            return
        for item in measurements:
            context = {
                "rollout_id": item.rollout_id,
                "edge_id": item.edge_id,
                "checkpoint_identity": item.before.model_identity,
                "estimator_version": item.before.estimator_version,
                "prompt_version": item.before.prompt_provenance.get("version", "unknown"),
            }
            values = {
                "uncertainty/semantic_entropy_before": item.before.entropy,
                "uncertainty/semantic_entropy_after": item.after.entropy,
                "uncertainty/entropy_reduction": item.absolute_entropy_reduction,
                "uncertainty/semantic_distribution_shift": item.semantic_distribution_shift,
                "uncertainty/cluster_count_before": item.before.cluster_count,
                "uncertainty/cluster_count_after": item.after.cluster_count,
                "uncertainty/sampling_seconds": item.sampling_seconds,
                "uncertainty/equivalence_seconds": item.equivalence_seconds,
            }
            if item.normalized_entropy_reduction is not None:
                values["uncertainty/normalized_entropy_reduction"] = (
                    item.normalized_entropy_reduction
                )
            for name, value in values.items():
                self.metric_sink.write(
                    MetricObservation(name=name, value=float(value), step=step, context=context)
                )

    def verbose_print(self, message: str) -> None:
        if self.verbose:
            print(f"[train] {message}", flush=True)


def score_selected_tokens(
    *,
    student: TrainableStudent,
    attempts: tuple[AnnotatedAttempt, ...],
    selections: dict[str, TokenSelection],
) -> StudentPredictionBatch:
    torch = __import__("torch")
    scores: dict[str, Any] = {}
    for attempt in attempts:
        logits = []
        for selected_generation in selections[attempt.rollout_id].generations:
            generation = next(
                item
                for item in attempt.annotations.generations
                if item.generation_id == selected_generation.generation_id
            )
            generated_text = GeneratedText(
                text=generation.text,
                prompt_token_ids=generation.prompt_token_ids,
                token_ids=generation.token_ids,
                token_offsets=generation.token_offsets,
                student=student.model_info,
                tokenizer=student.tokenizer_info,
            )
            predictions = student.score_tokens(
                generated_text,
                with_gradients=True,
                return_logits=True,
                return_logprobs=False,
                positions=selected_generation.positions,
            )
            if predictions.logits is None or not bool(predictions.logits.requires_grad):
                raise ValueError("student logits must be present and retain gradients")
            logits.append(predictions.logits)
        if not logits:
            raise ValueError("student scoring received an empty token selection")
        scores[attempt.rollout_id] = torch.cat(logits, dim=0)
    return StudentPredictionBatch(logits=scores)


def update_student(
    *,
    optimizer: Any,
    parameters: tuple[Any, ...],
    max_gradient_norm: float,
    gradient_scaler: Any | None,
) -> float:
    torch = __import__("torch")
    if gradient_scaler is not None:
        gradient_scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_gradient_norm,
        error_if_nonfinite=True,
    )
    value = finite_scalar(norm, "gradient norm")
    if gradient_scaler is not None:
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
    else:
        optimizer.step()
    return value


def finite_scalar(value: Any, name: str) -> float:
    scalar = (
        float(value.detach().float().cpu().item())
        if hasattr(value, "numel") and value.numel() == 1
        else float(value)
    )
    if not math.isfinite(scalar):
        raise FloatingPointError(f"{name} is non-finite")
    return scalar


def feedback_record(bundle: FeedbackBundle) -> FeedbackRecord:
    return FeedbackRecord(
        environment={item.feedback_id: item.model_dump(mode="json") for item in bundle.environment},
        judge_assessments=tuple(item.model_dump(mode="json") for item in bundle.local_assessments),
        projections=tuple(item.model_dump(mode="json") for item in bundle.projections),
        overall_assessment=(
            bundle.overall_assessment.model_dump(mode="json")
            if bundle.overall_assessment is not None
            else {}
        ),
        uncertainty_assessments=tuple(
            item.model_dump(mode="json") for item in bundle.uncertainty_assessments
        ),
    )


__all__ = [
    "TrainingLoop",
    "TrainingMethod",
    "TrainingResult",
    "score_selected_tokens",
    "update_student",
]
