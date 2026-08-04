"""Verify immutable experiment composition, validation, and ablation presets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rlm_train.experiment import (
    ABLATION_PRESETS,
    DiagnosticsConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeedbackConfig,
    TeacherConfig,
    TrainingAlgorithm,
    TrainingConfig,
    resolve_ablation_preset,
)
from rlm_train.judge import TeacherFeedbackMode
from rlm_train.regularization import GramAnchorConfig
from rlm_train.sdpo import SDPOConfig, TeacherStrategy


def test_every_initial_ablation_resolves_to_explicit_component_fields():
    resolved = {name: resolve_ablation_preset(name) for name in ABLATION_PRESETS}

    assert resolved["base"].training.algorithm is TrainingAlgorithm.NONE
    assert resolved["grpo"].training.algorithm is TrainingAlgorithm.GRPO
    assert resolved["conventional_sdpo"].training.feedback.mode is TeacherFeedbackMode.FACTUAL
    assert resolved["moving_teacher_sdpo"].training.teacher.strategy is TeacherStrategy.EMA
    assert resolved["restricted_sdpo"].training.feedback.mode is TeacherFeedbackMode.SCALAR
    assert resolved["edge_local_sdpo"].training.feedback.mode is TeacherFeedbackMode.DIAGNOSTIC
    assert resolved["proposed_method"].training.gram.is_active
    assert all(
        config.fingerprint == ExperimentConfig.model_validate(config.resolved_dict()).fingerprint
        for config in resolved.values()
    )


def test_incompatible_algorithm_teacher_and_feedback_combinations_fail_fast():
    with pytest.raises(ValidationError, match="non-SDPO algorithms cannot activate a teacher"):
        TrainingConfig(algorithm="grpo", sdpo=None)
    with pytest.raises(ValidationError, match="SDPO requires a fixed or EMA teacher"):
        TrainingConfig(
            teacher=TeacherConfig(strategy="none", checkpoint_identity=None),
            feedback=FeedbackConfig(mode=TeacherFeedbackMode.DIAGNOSTIC),
            sdpo=SDPOConfig(),
        )
    with pytest.raises(ValidationError, match="fixed teachers do not accept"):
        TeacherConfig(strategy="fixed", ema_update_rate=0.1)
    with pytest.raises(ValidationError, match="EMA teachers require"):
        TeacherConfig(strategy="ema")


def test_observer_configuration_cannot_change_training_fingerprint():
    base = resolve_ablation_preset("edge_local_sdpo")
    disabled_observers = ExperimentConfig(
        training=base.training,
        evaluation=EvaluationConfig(
            diagnostics=DiagnosticsConfig(
                epistemic_markers=False,
                reasoning_dynamics=False,
                divergence=False,
                gram_drift=False,
            )
        ),
    )

    assert base.training_fingerprint == disabled_observers.training_fingerprint
    assert base.fingerprint != disabled_observers.fingerprint


def test_download_free_dry_run_configuration_loads_and_round_trips(tmp_path):
    source = Path("training/configs/ood-robust-synthetic.toml")
    config = ExperimentConfig.from_file(source)
    destination = tmp_path / "resolved.json"

    config.write_resolved(destination)
    reloaded = ExperimentConfig.from_file(destination)

    assert reloaded == config
    assert config.training.teacher.strategy is TeacherStrategy.FIXED
    assert config.training.feedback.mode is TeacherFeedbackMode.DIAGNOSTIC
    assert config.evaluation.benchmarks[0].adapter == "jsonl"


def test_non_sdpo_policy_configuration_is_explicitly_teacher_free():
    config = TrainingConfig(
        algorithm=TrainingAlgorithm.GRPO,
        teacher=TeacherConfig(strategy=TeacherStrategy.NONE, checkpoint_identity=None),
        feedback=FeedbackConfig(mode=None),
        sdpo=None,
        gram=GramAnchorConfig(),
    )

    assert config.teacher.strategy is TeacherStrategy.NONE
    assert config.feedback.mode is None
