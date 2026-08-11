"""Explicitly construct the collaborators used by a training run."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm_train.attempts import create_attempt_runner
from rlm_train.datasets.build import build_dataset
from rlm_train.feedback import FeedbackCollector
from rlm_train.judge.create_judge import create_judge
from rlm_train.metrics.build import build_metric_sink
from rlm_train.saved_runs import CheckpointWriter, TrainingRecordWriter
from rlm_train.sdpo.calculate_loss import calculate_loss as calculate_sdpo_loss
from rlm_train.settings import RunSettings
from rlm_train.student import TrainableStudent, create_student
from rlm_train.training.optimizer import create_optimizer
from rlm_train.training.requirements import TrainingRequirements
from rlm_train.training.scheduler import create_scheduler
from rlm_train.training.training_loop import TrainingLoop


@dataclass(frozen=True)
class SDPOTrainingMethod:
    settings: Any
    name: str = "sdpo"

    @property
    def weight(self) -> float:
        return float(self.settings.weight)

    @property
    def top_k(self) -> int:
        return int(self.settings.top_k)

    @property
    def requirements(self) -> TrainingRequirements:
        return TrainingRequirements(
            included_text=self.settings.token_scope,
            feedback_scope=self.settings.feedback_scope,
            feedback_predictions=True,
        )

    def calculate_loss(self, batch: Any) -> Any:
        return calculate_sdpo_loss(batch)


def create_training_methods(settings: Any) -> tuple[SDPOTrainingMethod, ...]:
    methods = []
    if settings.sdpo.enabled:
        methods.append(SDPOTrainingMethod(settings.sdpo))
    if settings.grpo.enabled:
        raise NotImplementedError("GRPO training-run construction is not implemented")
    if settings.gram.enabled:
        raise NotImplementedError("Gram training-run construction is not implemented")
    if not methods:
        raise ValueError("at least one training method must be enabled")
    return tuple(methods)


def precision_context(
    precision: str,
) -> Callable[[], AbstractContextManager[Any]]:
    def create_context() -> AbstractContextManager[Any]:
        if precision in {"bf16", "fp16"}:
            import torch

            if torch.cuda.is_available():
                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    return create_context


def create_training_run(
    settings: RunSettings,
    *,
    student: TrainableStudent,
    judge: Any,
    resume_checkpoint: str | Path | None = None,
    verbose: bool = False,
) -> TrainingLoop:
    """Construct the direct collaborators in the same order used by the loop."""
    if settings.training_dataset is None:
        raise ValueError("training_dataset must be set to create a training run")
    parameters = tuple(student.trainable_parameters())
    optimizer = create_optimizer(parameters, settings.runtime)
    scheduler = create_scheduler(
        optimizer,
        settings.runtime,
        total_steps=settings.runtime.max_optimizer_steps,
    )
    checkpoint_writer = CheckpointWriter(
        settings.artifacts.output_directory,
        policy=student,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_interval=settings.artifacts.checkpoint_interval,
        retain_checkpoints=settings.artifacts.retain_checkpoints,
        save_final_checkpoint=settings.artifacts.save_final_checkpoint,
    )
    initial_state = None
    if resume_checkpoint is not None:
        initial_state = checkpoint_writer.restore_training_state(
            resume_checkpoint,
            run_spec_fingerprint=settings.fingerprint,
        )
    attempt_writer = (
        None
        if settings.artifacts.rollout_json == "none"
        else TrainingRecordWriter(
            Path(settings.artifacts.output_directory) / "attempts",
            mode=settings.artifacts.rollout_json,
        )
    )
    return TrainingLoop(
        run_settings=settings,
        dataset=build_dataset(settings.training_dataset),
        attempt_runner=create_attempt_runner(settings, student_client=student),
        feedback_collector=FeedbackCollector(judge),
        student=student,
        training_methods=create_training_methods(settings.objectives),
        optimizer=optimizer,
        scheduler=scheduler,
        attempt_writer=attempt_writer,
        metric_sink=(
            build_metric_sink(settings.artifacts.output_directory)
            if settings.artifacts.metrics_jsonl
            else None
        ),
        checkpoint_writer=checkpoint_writer,
        initial_state=initial_state,
        precision_context=precision_context(settings.runtime.precision),
        verbose=verbose,
    )


def create_default_training_run(
    settings: RunSettings,
    *,
    checkpoint_path: str | Path | None = None,
    resume_training: bool = False,
    verbose: bool = False,
) -> TrainingLoop:
    student = create_student(
        settings.student,
        runtime=settings.runtime,
        checkpoint_path=checkpoint_path,
    )
    judge = create_judge(settings.judge)
    return create_training_run(
        settings,
        student=student,
        judge=judge,
        resume_checkpoint=checkpoint_path if resume_training else None,
        verbose=verbose,
    )


__all__ = [
    "SDPOTrainingMethod",
    "create_default_training_run",
    "create_training_methods",
    "create_training_run",
]
