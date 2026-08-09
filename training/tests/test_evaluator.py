"""End-to-end wiring of the recursive evaluator and gradable predictions writer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from rlm_train.datasets.records import DatasetRecord
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.rollouts.protocol import RolloutRequest, RolloutResult
from rlm_train.runtime import ComponentFactory, register_evaluator_builder
from rlm_train.spec.artifacts import ArtifactSpec
from rlm_train.spec.models import StudentSpec
from rlm_train.spec.run import RunSpec
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    ExecutionNode,
    ExecutionRecord,
    NodeRole,
    TaskPartition,
)


class FakeDataset:
    identity = "held-out-v1"

    def __init__(self, records: tuple[DatasetRecord, ...]) -> None:
        self._records = records

    def records(self) -> tuple[DatasetRecord, ...]:
        return self._records


class EchoRolloutEngine:
    def execute(self, request: RolloutRequest) -> RolloutResult:
        answer = f"answer::{request.task_id}"
        completion = SimpleNamespace(response=answer)
        rollout = AnnotatedRollout(
            rollout_id=f"rollout-{request.task_id}",
            mode=request.mode,
            task=TaskPartition(task_id=request.task_id, public=request.public_task),
            policy={"policy_owner": "student"},
            execution=ExecutionRecord(
                root_node_id="root",
                nodes=(
                    ExecutionNode(
                        node_id="root",
                        role=NodeRole.ROOT,
                        depth=0,
                        prompt=request.public_task,
                        result=answer,
                    ),
                ),
                events=({"sequence_number": 0},),
            ),
            result={"final_answer": answer},
        )
        return RolloutResult(completion=completion, rollout=rollout)


class LengthScorer:
    def score(self, record: DatasetRecord, final_answer: str) -> tuple[float, dict[str, Any]]:
        return float(len(final_answer)), {"length": len(final_answer)}


def held_out_records() -> tuple[DatasetRecord, ...]:
    return (
        DatasetRecord(record_id="q1", public_task={"prompt": "first question"}),
        DatasetRecord(record_id="q2", public_task={"prompt": "second question"}),
    )


def test_evaluator_writes_gradable_predictions(tmp_path):
    evaluator = RecursiveEvaluator(
        dataset=FakeDataset(held_out_records()),
        rollout_engine=EchoRolloutEngine(),
        scorer=LengthScorer(),
        output_directory=tmp_path,
        checkpoint_id="ckpt-1",
    )

    results = evaluator.evaluate()

    assert [record.record_id for record in results] == ["q1", "q2"]
    assert results[0].final_answer == "answer::q1"

    predictions_path = tmp_path / "predictions.jsonl"
    assert predictions_path.exists()
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert rows[0]["record_id"] == "q1"
    assert rows[0]["final_answer"] == "answer::q1"
    assert rows[0]["question"] == "first question"
    assert rows[0]["checkpoint_id"] == "ckpt-1"
    assert rows[0]["score"] == len("answer::q1")


def test_evaluator_builder_resolves_through_factory(tmp_path):
    spec = RunSpec(
        student=StudentSpec(model_id="student"),
        artifacts=ArtifactSpec(output_directory=str(tmp_path)),
    )

    factory = ComponentFactory()
    register_evaluator_builder(
        factory,
        dataset=FakeDataset(held_out_records()),
        rollout_engine=EchoRolloutEngine(),
        scorer=LengthScorer(),
        checkpoint_id="ckpt-2",
    )
    resolved = factory.resolve(spec)

    assert resolved.evaluator is not None
    results = resolved.evaluator.evaluate()
    assert len(results) == 2
    assert (tmp_path / "predictions.jsonl").exists()
