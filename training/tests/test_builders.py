"""Per-package build entry points return correctly typed, spec-driven components."""

from __future__ import annotations

import json

import pytest

from rlm_train.artifacts.build import build_artifact_writer
from rlm_train.artifacts.rollout_json import RolloutJSONWriter
from rlm_train.datasets.adapters.hotpotqa import HotpotQADataset
from rlm_train.datasets.adapters.jsonl import JSONLDataset
from rlm_train.datasets.build import build_dataset
from rlm_train.engine.providers import (
    JudgeFeedbackProvider,
    SelfDistillationTeacherTargetProvider,
    TransformersPolicyScoreProvider,
    build_feedback_provider,
    build_policy_score_provider,
)
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.evaluation.scorers import build_scorer
from rlm_train.judge.providers.fake import DeterministicFakeJudge
from rlm_train.metrics.build import build_metric_sink
from rlm_train.metrics.jsonl import JSONLMetricSink
from rlm_train.rollouts.build import build_rollout_engine
from rlm_train.rollouts.rlm_engine import RLMRolloutEngine
from rlm_train.spec.models import StudentSpec, TeacherSpec, TeacherStrategy
from rlm_train.spec.run import DatasetRefSpec, RunSpec, RuntimeSpec
from rlm_train.teachers.build import build_teacher_target_provider, build_teachers


class FakePolicy:
    model_name = "student"

    def trainable_parameters(self):
        return ()


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_build_dataset_returns_jsonl_dataset(tmp_path):
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [{"id": "a", "question": "q", "context": "evidence", "target": "t"}],
    )
    dataset = build_dataset(DatasetRefSpec(source=str(path)))
    assert isinstance(dataset, JSONLDataset)
    records = dataset.records()
    assert [record.record_id for record in records] == ["a"]
    assert records[0].public_task == {"question": "q", "context": "evidence"}


def test_jsonl_dataset_rejects_combined_legacy_prompt(tmp_path):
    path = write_jsonl(tmp_path / "legacy.jsonl", [{"id": "a", "prompt": "context + q"}])

    with pytest.raises(ValueError, match="keep question and context separate"):
        JSONLDataset(path).records()


def test_build_dataset_rejects_unknown_adapter(tmp_path):
    with pytest.raises(ValueError, match="unsupported dataset adapter"):
        build_dataset(DatasetRefSpec(adapter="parquet", source=str(tmp_path / "x")))


def test_build_dataset_returns_configured_hotpotqa_dataset():
    dataset = build_dataset(
        DatasetRefSpec(
            adapter="hotpotqa",
            source="hotpotqa/hotpot_qa",
            subset="distractor",
            split="train",
            revision="revision-1",
            max_records=200,
        )
    )

    assert isinstance(dataset, HotpotQADataset)
    assert dataset.repository == "hotpotqa/hotpot_qa"
    assert dataset.subset == "distractor"
    assert dataset.split == "train"
    assert dataset.revision == "revision-1"
    assert dataset.max_records == 200


def test_build_optimizer_requires_parameters():
    pytest.importorskip("torch")
    from rlm_train.engine.optimizer import build_optimizer

    with pytest.raises(ValueError, match="at least one trainable parameter"):
        build_optimizer([], RuntimeSpec())


def test_build_optimizer_creates_adamw_over_parameters():
    torch = pytest.importorskip("torch")
    from rlm_train.engine.optimizer import build_optimizer

    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = build_optimizer([parameter], RuntimeSpec(learning_rate=0.01))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_build_metric_sink_and_artifact_writer(tmp_path):
    assert isinstance(build_metric_sink(tmp_path), JSONLMetricSink)
    assert isinstance(build_artifact_writer(tmp_path), RolloutJSONWriter)
    assert build_artifact_writer(tmp_path, mode="none") is None


def test_build_rollout_engine_from_spec():
    spec = RunSpec(student=StudentSpec(model_id="student", policy_owner="student"))
    engine = build_rollout_engine(spec, policy=FakePolicy())
    assert isinstance(engine, RLMRolloutEngine)


def test_build_providers_return_expected_types():
    policy = FakePolicy()
    assert isinstance(build_policy_score_provider(policy), TransformersPolicyScoreProvider)
    assert isinstance(build_feedback_provider(DeterministicFakeJudge()), JudgeFeedbackProvider)


def test_build_teacher_target_provider_current_policy():
    provider = build_teacher_target_provider(TeacherSpec(), policy=FakePolicy(), top_k=8)
    assert isinstance(provider, SelfDistillationTeacherTargetProvider)
    assert build_teachers(TeacherSpec(), policy=FakePolicy()) == ()


def test_build_teacher_target_provider_rejects_unwired_strategy():
    spec = TeacherSpec(strategy=TeacherStrategy.FIXED, model_id="teacher")
    with pytest.raises(NotImplementedError):
        build_teacher_target_provider(spec, policy=FakePolicy(), top_k=8)


def test_build_evaluator_from_spec(tmp_path):
    eval_path = write_jsonl(
        tmp_path / "eval.jsonl",
        [{"id": "e1", "question": "q", "context": "evidence", "target": "t"}],
    )
    from rlm_train.evaluation.build import build_evaluator

    spec = RunSpec(
        student=StudentSpec(model_id="student", policy_owner="student"),
        evaluation_datasets=(DatasetRefSpec(source=str(eval_path), split="test"),),
    )
    evaluator = build_evaluator(spec, policy=FakePolicy(), scorer=build_scorer("exact_match"))
    assert isinstance(evaluator, RecursiveEvaluator)


def test_runtime_spec_rejects_warmup_exceeding_steps():
    with pytest.raises(ValueError, match="warmup_steps cannot exceed"):
        RuntimeSpec(max_optimizer_steps=5, warmup_steps=10)
