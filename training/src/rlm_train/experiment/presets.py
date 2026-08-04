"""Resolve the initial ablation matrix into explicit experiment component fields."""

from __future__ import annotations

from rlm_train.experiment.config import (
    EvaluationConfig,
    ExperimentConfig,
    FeedbackConfig,
    TeacherConfig,
    TrainingAlgorithm,
    TrainingConfig,
)
from rlm_train.judge import TeacherFeedbackMode
from rlm_train.regularization import GramAnchorConfig, GramAnchorSourceConfig
from rlm_train.sdpo import SDPOConfig, TeacherStrategy

ABLATION_PRESETS = (
    "base",
    "grpo",
    "conventional_sdpo",
    "moving_teacher_sdpo",
    "restricted_sdpo",
    "edge_local_sdpo",
    "proposed_method",
)


def resolve_ablation_preset(
    name: str,
    *,
    evaluation: EvaluationConfig | None = None,
    anchor_checkpoint_identity: str = "initial-policy",
) -> ExperimentConfig:
    """Return a complete immutable configuration for one named initial arm."""
    if name not in ABLATION_PRESETS:
        raise ValueError(f"unknown ablation preset {name!r}")
    resolved_evaluation = evaluation or EvaluationConfig()
    if name == "base":
        training = _policy_only(TrainingAlgorithm.NONE)
    elif name == "grpo":
        training = _policy_only(TrainingAlgorithm.GRPO)
    else:
        strategy = TeacherStrategy.EMA if name == "moving_teacher_sdpo" else TeacherStrategy.FIXED
        update_rate = 0.05 if strategy is TeacherStrategy.EMA else None
        mode = {
            "conventional_sdpo": TeacherFeedbackMode.FACTUAL,
            "moving_teacher_sdpo": TeacherFeedbackMode.FACTUAL,
            "restricted_sdpo": TeacherFeedbackMode.SCALAR,
            "edge_local_sdpo": TeacherFeedbackMode.DIAGNOSTIC,
            "proposed_method": TeacherFeedbackMode.DIAGNOSTIC,
        }[name]
        gram = GramAnchorConfig()
        if name == "proposed_method":
            gram = GramAnchorConfig(
                enabled=True,
                loss_weight=0.1,
                anchor=GramAnchorSourceConfig(checkpoint_path=anchor_checkpoint_identity),
            )
        training = TrainingConfig(
            algorithm=TrainingAlgorithm.SDPO,
            teacher=TeacherConfig(
                strategy=strategy,
                checkpoint_identity="initial-policy",
                ema_update_rate=update_rate,
            ),
            feedback=FeedbackConfig(mode=mode, projector_version="v1"),
            sdpo=SDPOConfig(teacher=strategy, ema_update_rate=update_rate),
            gram=gram,
        )
    return ExperimentConfig(training=training, evaluation=resolved_evaluation)


def _policy_only(algorithm: TrainingAlgorithm) -> TrainingConfig:
    """Construct an explicit no-teacher, no-feedback baseline configuration."""
    return TrainingConfig(
        algorithm=algorithm,
        teacher=TeacherConfig(
            strategy=TeacherStrategy.NONE,
            checkpoint_identity=None,
            ema_update_rate=None,
        ),
        feedback=FeedbackConfig(mode=None),
        sdpo=None,
        gram=GramAnchorConfig(),
    )


__all__ = ["ABLATION_PRESETS", "resolve_ablation_preset"]
