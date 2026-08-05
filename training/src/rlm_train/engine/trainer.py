"""Canonical, objective-driven training orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rlm_train.datasets.records import DatasetRecord
from rlm_train.feedback.schema import FeedbackBundle
from rlm_train.metrics.schema import MetricObservation
from rlm_train.objectives.composer import ComposedObjectiveResult, ObjectiveComposer
from rlm_train.objectives.protocol import ObjectiveBatch, ObjectiveCapabilities
from rlm_train.rollouts.protocol import RolloutEngine, RolloutRequest
from rlm_train.rollouts.selectors import TokenSelectionResult, select_tokens
from rlm_train.spec import RunSpec
from rlm_train.spec.feedback import AssessmentScope
from rlm_train.teachers.protocol import Teacher
from rlm_train.teachers.targets import TeacherTarget
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    FeedbackRecord,
)

from .batch import BatchRequirements, plan_batch_requirements
from .state import TrainerState


class Trainer(Protocol):
    def train(self) -> Any: ...


class DatasetSource(Protocol):
    def records(self) -> Sequence[DatasetRecord]: ...


class RewardProvider(Protocol):
    """Compute verifier-owned rewards without exposing private data to the policy."""

    def score(
        self,
        record: DatasetRecord,
        rollouts: tuple[AnnotatedRollout, ...],
    ) -> RewardBatch: ...


class FeedbackProvider(Protocol):
    """Construct only the judge scopes requested by enabled objectives."""

    def assess(
        self,
        record: DatasetRecord,
        rollouts: tuple[AnnotatedRollout, ...],
        scopes: frozenset[AssessmentScope],
    ) -> FeedbackBundle: ...


class TeacherTargetProvider(Protocol):
    """Build detached targets for the exact selected sampled tokens."""

    def build(
        self,
        objective: str,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, TokenSelectionResult],
        feedback: FeedbackBundle,
    ) -> dict[str, TeacherTarget]: ...


class PolicyScoreProvider(Protocol):
    """Recompute differentiable scores for exact sampled IDs."""

    def score(
        self,
        objective: str,
        capabilities: ObjectiveCapabilities,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, TokenSelectionResult],
    ) -> PolicyScoreBatch: ...


class ArtifactWriter(Protocol):
    def write(self, rollout: AnnotatedRollout) -> Path: ...


class MetricRecorder(Protocol):
    def record(self, observation: MetricObservation) -> None: ...


class MetricSink(Protocol):
    def write(self, observation: MetricObservation) -> None: ...


@dataclass(frozen=True)
class RewardBatch:
    rewards: dict[str, float]
    advantages: dict[str, float]


@dataclass(frozen=True)
class PolicyScoreBatch:
    policy_scores: dict[str, Any]
    behavior_policy_scores: dict[str, Any] = field(default_factory=dict)
    hidden_states: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalTrainingResult:
    state: TrainerState
    final_loss: float
    rollout_artifacts: tuple[str, ...] = ()


class CanonicalTrainer:
    """Execute the standard full-RLM flow from declared objective capabilities."""

    def __init__(
        self,
        *,
        spec: RunSpec,
        dataset: DatasetSource,
        rollout_engine: RolloutEngine,
        objectives: ObjectiveComposer,
        optimizer: Any,
        policy_scores: PolicyScoreProvider,
        policy_owner: str,
        policy_parameters: Iterable[Any],
        rewards: RewardProvider | None = None,
        feedback: FeedbackProvider | None = None,
        teacher_targets: TeacherTargetProvider | None = None,
        teachers: Iterable[Teacher] = (),
        scheduler: Any | None = None,
        artifact_writer: ArtifactWriter | None = None,
        metric_recorder: MetricRecorder | None = None,
        metric_sink: MetricSink | None = None,
        precision_context: Callable[[], AbstractContextManager[Any]] | None = None,
        gradient_scaler: Any | None = None,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.dataset = dataset
        self.rollout_engine = rollout_engine
        self.objectives = objectives
        self.optimizer = optimizer
        self.policy_scores = policy_scores
        self.policy_owner = policy_owner
        self.policy_parameters = tuple(policy_parameters)
        self.rewards = rewards
        self.feedback = feedback
        self.teacher_targets = teacher_targets
        self.teachers = tuple(teachers)
        self.scheduler = scheduler
        self.artifact_writer = artifact_writer
        self.metric_recorder = metric_recorder
        self.metric_sink = metric_sink
        self.precision_context = precision_context or nullcontext
        self.gradient_scaler = gradient_scaler
        self.verbose = verbose

    def train(self) -> CanonicalTrainingResult:
        capabilities = self.objectives.capabilities
        requirements = plan_batch_requirements(capabilities)
        self._validate_requirements(requirements)
        records = tuple(self.dataset.records())
        if not records:
            raise ValueError("training dataset is empty")

        runtime = self.spec.runtime
        state = TrainerState(run_spec_fingerprint=self.spec.fingerprint)
        artifact_paths: list[str] = []
        final_loss = 0.0
        record_index = 0

        self._verbose_print(
            "starting training: "
            f"records={len(records)} objectives={tuple(capabilities)} "
            f"optimizer_steps={runtime.max_optimizer_steps} "
            f"gradient_accumulation_steps={runtime.gradient_accumulation_steps}"
        )

        for optimizer_step in range(1, runtime.max_optimizer_steps + 1):
            self._verbose_print(
                f"optimizer step {optimizer_step}/{runtime.max_optimizer_steps} started"
            )
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for accumulation_step in range(1, runtime.gradient_accumulation_steps + 1):
                record = records[record_index % len(records)]
                record_index += 1
                self._verbose_print(
                    f"  accumulation {accumulation_step}/{runtime.gradient_accumulation_steps} "
                    f"record={record.record_id}"
                )
                rollouts = self._execute_rollouts(record, requirements.rollout_count)
                self._verbose_print(
                    f"  generated {len(rollouts)} rollout(s) for record={record.record_id}"
                )
                selection_batches = self._select_objective_tokens(rollouts, capabilities)
                rollouts = self._attach_selections(rollouts, selection_batches)
                reward_batch = self._prepare_rewards(record, rollouts, requirements)
                feedback_bundle = self._prepare_feedback(record, rollouts, requirements)

                with self.precision_context():
                    batches, targets = self._prepare_objective_batches(
                        rollouts,
                        selection_batches,
                        capabilities,
                        reward_batch,
                        feedback_bundle,
                    )
                    composed = self.objectives.compute(batches)
                    scaled_loss = composed.loss / runtime.gradient_accumulation_steps
                loss_value = _finite_scalar(composed.loss, "composed loss")
                self._verbose_print(f"  computed loss={loss_value:.6f}")
                self._backward(scaled_loss)
                accumulated_loss += loss_value / runtime.gradient_accumulation_steps
                artifact_paths.extend(
                    self._write_rollouts(
                        rollouts,
                        feedback_bundle,
                        targets,
                        composed,
                    )
                )

            gradient_norm = self._optimizer_step(runtime.max_gradient_norm)
            if self.scheduler is not None:
                self.scheduler.step()
            for teacher in self.teachers:
                teacher.after_optimizer_step()

            final_loss = accumulated_loss
            state = TrainerState(
                optimizer_step=optimizer_step,
                examples_seen=record_index,
                run_spec_fingerprint=self.spec.fingerprint,
            )
            self._record_metrics(optimizer_step, final_loss, gradient_norm)
            self._verbose_print(
                f"optimizer step {optimizer_step}/{runtime.max_optimizer_steps} complete: "
                f"loss={final_loss:.6f} gradient_norm={gradient_norm:.6f} "
                f"examples_seen={record_index} artifacts={len(artifact_paths)}"
            )

        result = CanonicalTrainingResult(
            state=state,
            final_loss=final_loss,
            rollout_artifacts=tuple(artifact_paths),
        )
        self._verbose_print(
            f"training complete: final_loss={result.final_loss:.6f} "
            f"optimizer_step={result.state.optimizer_step} "
            f"examples_seen={result.state.examples_seen} "
            f"artifacts={len(result.rollout_artifacts)}"
        )
        return result

    def _validate_requirements(self, requirements: BatchRequirements) -> None:
        if requirements.rewards and self.rewards is None:
            raise ValueError("enabled objectives require a reward provider")
        if requirements.feedback_scopes and self.feedback is None:
            raise ValueError("enabled objectives require a scoped feedback provider")
        if requirements.teacher_targets and self.teacher_targets is None:
            raise ValueError("enabled objectives require a teacher-target provider")

    def _execute_rollouts(
        self,
        record: DatasetRecord,
        count: int,
    ) -> tuple[AnnotatedRollout, ...]:
        rollouts = tuple(
            self.rollout_engine.execute(
                RolloutRequest(
                    task_id=record.record_id,
                    public_task=record.public_task,
                    private_reference=record.verifier_data,
                    mode="training",
                )
            ).rollout
            for _ in range(count)
        )
        for rollout in rollouts:
            if rollout.policy.get("policy_owner") != self.policy_owner:
                raise ValueError("rollout was not generated by the configured trainable policy")
        return rollouts

    def _select_objective_tokens(
        self,
        rollouts: tuple[AnnotatedRollout, ...],
        capabilities: dict[str, ObjectiveCapabilities],
    ) -> dict[str, dict[str, TokenSelectionResult]]:
        selected: dict[str, dict[str, TokenSelectionResult]] = {}
        for objective, capability in capabilities.items():
            objective_selections = {
                rollout.rollout_id: select_tokens(
                    rollout,
                    objective=objective,
                    token_scope=capability.token_scope,
                    policy_owner=self.policy_owner,
                )
                for rollout in rollouts
            }
            if any(item.durable.active_token_count == 0 for item in objective_selections.values()):
                raise ValueError(f"{objective} selected no tokens for at least one rollout")
            selected[objective] = objective_selections
        return selected

    def _attach_selections(
        self,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, dict[str, TokenSelectionResult]],
    ) -> tuple[AnnotatedRollout, ...]:
        enriched: list[AnnotatedRollout] = []
        for rollout in rollouts:
            values = {
                name: selected[rollout.rollout_id].durable for name, selected in selections.items()
            }
            annotations = rollout.annotations.model_copy(update={"objective_selections": values})
            enriched.append(rollout.model_copy(update={"annotations": annotations}))
        return tuple(enriched)

    def _prepare_rewards(
        self,
        record: DatasetRecord,
        rollouts: tuple[AnnotatedRollout, ...],
        requirements: BatchRequirements,
    ) -> RewardBatch:
        if not requirements.rewards:
            return RewardBatch(rewards={}, advantages={})
        assert self.rewards is not None
        result = self.rewards.score(record, rollouts)
        expected = {rollout.rollout_id for rollout in rollouts}
        if set(result.rewards) != expected or set(result.advantages) != expected:
            raise ValueError("reward and advantage maps must cover every rollout exactly")
        for value in (*result.rewards.values(), *result.advantages.values()):
            if not math.isfinite(value):
                raise FloatingPointError("reward provider returned a non-finite value")
        return result

    def _prepare_feedback(
        self,
        record: DatasetRecord,
        rollouts: tuple[AnnotatedRollout, ...],
        requirements: BatchRequirements,
    ) -> FeedbackBundle:
        if not requirements.feedback_scopes:
            return FeedbackBundle()
        assert self.feedback is not None
        return self.feedback.assess(record, rollouts, requirements.feedback_scopes)

    def _prepare_objective_batches(
        self,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, dict[str, TokenSelectionResult]],
        capabilities: dict[str, ObjectiveCapabilities],
        rewards: RewardBatch,
        feedback: FeedbackBundle,
    ) -> tuple[dict[str, ObjectiveBatch], dict[str, dict[str, TeacherTarget]]]:
        batches: dict[str, ObjectiveBatch] = {}
        all_targets: dict[str, dict[str, TeacherTarget]] = {}
        for objective, capability in capabilities.items():
            selected = selections[objective]
            score_batch = self.policy_scores.score(
                objective,
                capability,
                rollouts,
                selected,
            )
            if capability.behavior_logprobs and not score_batch.behavior_policy_scores:
                raise ValueError(f"{objective} requires behavior-policy log probabilities")
            if capability.hidden_states and not score_batch.hidden_states:
                raise ValueError(f"{objective} requires hidden states")
            targets: dict[str, TeacherTarget] = {}
            if capability.teacher_targets:
                assert self.teacher_targets is not None
                targets = self.teacher_targets.build(
                    objective,
                    rollouts,
                    selected,
                    feedback,
                )
                if set(targets) != {rollout.rollout_id for rollout in rollouts}:
                    raise ValueError(f"{objective} teacher targets must cover every rollout")
            all_targets[objective] = targets
            batches[objective] = ObjectiveBatch(
                rollouts=rollouts,
                token_selections={
                    rollout_id: selection.durable for rollout_id, selection in selected.items()
                },
                policy_scores=score_batch.policy_scores,
                behavior_policy_scores=score_batch.behavior_policy_scores,
                rewards=rewards.rewards if capability.rewards else {},
                advantages=rewards.advantages if capability.rewards else {},
                feedback=feedback if capability.feedback_scope is not None else None,
                teacher_targets=targets,
                hidden_states=score_batch.hidden_states,
            )
        return batches, all_targets

    def _backward(self, loss: Any) -> None:
        _finite_scalar(loss, "scaled loss")
        if self.gradient_scaler is not None:
            self.gradient_scaler.scale(loss).backward()
            return
        if not hasattr(loss, "backward"):
            raise TypeError("composed training loss must support backward()")
        loss.backward()

    def _optimizer_step(self, max_gradient_norm: float) -> float:
        torch = __import__("torch")
        if self.gradient_scaler is not None:
            self.gradient_scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_parameters,
            max_gradient_norm,
            error_if_nonfinite=True,
        )
        gradient_norm_value = _finite_scalar(gradient_norm, "gradient norm")
        if self.gradient_scaler is not None:
            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()
        else:
            self.optimizer.step()
        return gradient_norm_value

    def _write_rollouts(
        self,
        rollouts: tuple[AnnotatedRollout, ...],
        feedback: FeedbackBundle,
        targets: dict[str, dict[str, TeacherTarget]],
        composed: ComposedObjectiveResult,
    ) -> list[str]:
        if self.artifact_writer is None:
            return []
        paths: list[str] = []
        for rollout in rollouts:
            rollout_targets = tuple(
                target.model_dump(mode="json")
                for values in targets.values()
                for rollout_id, target in values.items()
                if rollout_id == rollout.rollout_id
            )
            objective_records = {
                name: {
                    "active_token_count": result.active_token_count,
                    "diagnostics": result.diagnostics,
                    "attributions": [
                        attribution.__dict__
                        for attribution in result.attributions
                        if attribution.rollout_id == rollout.rollout_id
                    ],
                }
                for name, result in composed.results.items()
            }
            enriched = rollout.model_copy(
                update={
                    "feedback": _feedback_record(feedback),
                    "teacher_targets": rollout_targets,
                    "objectives": objective_records,
                }
            )
            paths.append(str(self.artifact_writer.write(enriched)))
        return paths

    def _record_metrics(self, step: int, loss: float, gradient_norm: float) -> None:
        observations = (
            MetricObservation(name="train/loss/total", value=loss, step=step),
            MetricObservation(
                name="train/optimizer/gradient_norm",
                value=gradient_norm,
                step=step,
            ),
        )
        for observation in observations:
            if self.metric_recorder is not None:
                self.metric_recorder.record(observation)
            if self.metric_sink is not None:
                self.metric_sink.write(observation)

    def _verbose_print(self, message: str) -> None:
        if self.verbose:
            print(f"[train] {message}", flush=True)


def _feedback_record(bundle: FeedbackBundle) -> FeedbackRecord:
    return FeedbackRecord(
        environment={item.feedback_id: item.model_dump(mode="json") for item in bundle.environment},
        judge_assessments=tuple(item.model_dump(mode="json") for item in bundle.local_assessments),
        projections=tuple(item.model_dump(mode="json") for item in bundle.projections),
        overall_assessment=(
            bundle.overall_assessment.model_dump(mode="json")
            if bundle.overall_assessment is not None
            else {}
        ),
    )


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise TypeError(f"{name} must be scalar")
        scalar = float(value.detach().float().cpu().item())
    else:
        scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"{name} is non-finite")
    return scalar


__all__ = [
    "CanonicalTrainer",
    "CanonicalTrainingResult",
    "FeedbackProvider",
    "PolicyScoreBatch",
    "PolicyScoreProvider",
    "RewardBatch",
    "RewardProvider",
    "TeacherTargetProvider",
    "Trainer",
]
