"""Assemble the configured fixed-teacher SDPO path without trainer-specific source edits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm_train.colab.config import ColabRunConfig
from rlm_train.colab.generation import PromptFormatter
from rlm_train.colab.teacher import (
    FileTeacherTargetCache,
    TransformersQuestionTeacherProvider,
    build_fixed_teacher_controller,
)
from rlm_train.colab.trainer import MaskedQuestionSDPOLossBuilder
from rlm_train.colab.trajectory_sdpo import TrajectoryQuestionTargetProvider
from rlm_train.experiment.config import TrainingAlgorithm
from rlm_train.judge import (
    DeterministicFakeStructuredJudgeClient,
    OpenAIStructuredJudgeClient,
    PrivilegedJudgeContext,
    SQLiteFeedbackCache,
    StructuredOutputTrajectoryJudge,
)
from rlm_train.sdpo import TeacherStrategy
from rlm_train.trajectory import TrajectoryCompiler


@dataclass(frozen=True)
class FixedSDPOComponents:
    """Return the runnable builder plus checkpoint/cache component handles."""

    loss_builder: MaskedQuestionSDPOLossBuilder
    judge: StructuredOutputTrajectoryJudge
    teacher: TransformersQuestionTeacherProvider
    judge_cache: SQLiteFeedbackCache
    teacher_cache: FileTeacherTargetCache


def build_fixed_sdpo_components(
    configuration: ColabRunConfig,
    *,
    student: Any,
    tokenizer: Any,
    tokenizer_fingerprint: str,
    output_directory: str | Path,
    privileged_contexts: Mapping[str, PrivilegedJudgeContext | None] | None = None,
) -> FixedSDPOComponents:
    """Resolve config-selected judge, projector, fixed teacher, masks, and caches."""
    experiment = configuration.resolved_experiment
    if experiment.training.algorithm is not TrainingAlgorithm.SDPO:
        raise ValueError("fixed SDPO assembly requires an SDPO experiment")
    if experiment.training.teacher.strategy is not TeacherStrategy.FIXED:
        raise ValueError("the initial single-GPU assembly supports only a fixed teacher")
    if experiment.training.feedback.mode is None or experiment.training.sdpo is None:
        raise ValueError("SDPO assembly requires feedback and SDPO component configuration")
    output = Path(output_directory)
    judge_cache = SQLiteFeedbackCache(output / "judge-feedback.sqlite3")
    teacher_cache = FileTeacherTargetCache(output / configuration.teacher_runtime.cache_directory)
    if configuration.judge.provider == "fake":
        client = DeterministicFakeStructuredJudgeClient()
    else:
        client = OpenAIStructuredJudgeClient(
            model=configuration.judge.model,
            model_revision=configuration.judge.model_revision,
            prompt_schema_version=configuration.judge.prompt_schema_version,
            api_key_environment=configuration.judge.api_key_environment,
        )
    judge_version = (
        f"{configuration.judge.provider}:{configuration.judge.model}:"
        f"{configuration.judge.model_revision}"
    )
    judge = StructuredOutputTrajectoryJudge(
        client,
        judge_version=judge_version,
        rubric_version=configuration.judge.prompt_schema_version,
        max_attempts=configuration.judge.max_attempts,
        cache=judge_cache,
    )
    checkpoint_identity = experiment.training.teacher.checkpoint_identity
    assert checkpoint_identity is not None
    controller = build_fixed_teacher_controller(
        student,
        checkpoint_identity=checkpoint_identity,
    )
    teacher = TransformersQuestionTeacherProvider(
        controller,
        tokenizer,
        PromptFormatter(tokenizer, configuration.generation),
        student_tokenizer_fingerprint=tokenizer_fingerprint,
        residency=configuration.teacher_runtime.residency,
        top_k=experiment.training.sdpo.top_k,
        cache=teacher_cache,
        student_model=student,
    )
    compiler = TrajectoryCompiler(
        feedback_mode=experiment.training.feedback.mode,
        projector_version=experiment.training.feedback.projector_version,
    )
    target_provider = TrajectoryQuestionTargetProvider(
        judge=judge,
        compiler=compiler,
        teacher=teacher,
        privileged_contexts=privileged_contexts,
    )
    return FixedSDPOComponents(
        loss_builder=MaskedQuestionSDPOLossBuilder(target_provider),
        judge=judge,
        teacher=teacher,
        judge_cache=judge_cache,
        teacher_cache=teacher_cache,
    )


__all__ = ["FixedSDPOComponents", "build_fixed_sdpo_components"]
