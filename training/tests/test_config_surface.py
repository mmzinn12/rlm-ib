"""Config surface: ExperimentSettings round-trip, scorer registry, and spec-driven assembly."""

from __future__ import annotations

import json

import pytest

from rlm_train.datasets.records import DatasetRecord
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.evaluation.scorers import ExactMatchScorer, build_scorer
from rlm_train.experiment import ExperimentSettings
from rlm_train.rollouts.rlm_engine import RLMRolloutEngine
from rlm_train.runtime import ComponentFactory, register_default_builders
from rlm_train.spec.artifacts import ArtifactSpec
from rlm_train.spec.models import StudentSpec
from rlm_train.spec.run import DatasetRefSpec, RunSpec


def test_experiment_settings_round_trip_through_json(tmp_path):
    settings = ExperimentSettings(
        scorer="exact_match",
        checkpoint_id="ckpt-3",
        render_predictions=False,
    )
    path = settings.write_json(tmp_path / "experiment.json")

    reloaded = ExperimentSettings.from_file(path)

    assert reloaded == settings
    assert reloaded.predictions_filename == "predictions.jsonl"


def test_experiment_settings_rejects_unknown_fields(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"scorer": "exact_match", "unknown": 1}))
    with pytest.raises(ValueError):
        ExperimentSettings.from_file(tmp_path / "bad.json")


def test_build_scorer_exact_match():
    scorer = build_scorer("exact_match")
    assert isinstance(scorer, ExactMatchScorer)

    record = DatasetRecord(record_id="q1", public_task={"prompt": "q"}, verifier_data="Paris")
    hit, detail = scorer.score(record, "  paris ")
    miss, _ = scorer.score(record, "London")

    assert hit == 1.0
    assert detail["correct"] is True
    assert miss == 0.0


def test_build_scorer_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorer("bleu")


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_register_default_builders_resolves_pipeline_from_spec(tmp_path):
    train_path = write_jsonl(
        tmp_path / "train.jsonl",
        [{"id": "t1", "prompt": "train question", "target": "a"}],
    )
    eval_path = write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {"id": "e1", "prompt": "first held-out", "target": "a"},
            {"id": "e2", "prompt": "second held-out", "target": "b"},
        ],
    )
    spec = RunSpec(
        student=StudentSpec(model_id="student"),
        training_dataset=DatasetRefSpec(source=str(train_path)),
        evaluation_datasets=(DatasetRefSpec(source=str(eval_path), split="test"),),
        artifacts=ArtifactSpec(output_directory=str(tmp_path / "out")),
    )

    class FakePolicy:
        model_name = "student"

    factory = ComponentFactory()
    register_default_builders(factory, policy=FakePolicy(), scorer=build_scorer("exact_match"))
    resolved = factory.resolve(spec)

    assert [record.record_id for record in resolved.dataset.records()] == ["t1"]
    assert isinstance(resolved.rollout_engine, RLMRolloutEngine)
    assert isinstance(resolved.evaluator, RecursiveEvaluator)
    # Held-out dataset is wired into the evaluator from the spec pointer.
    assert [record.record_id for record in resolved.evaluator.dataset.records()] == ["e1", "e2"]
