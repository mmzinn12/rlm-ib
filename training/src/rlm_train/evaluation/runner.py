"""Whole-recursive-student evaluation through the canonical attempt protocol."""

from __future__ import annotations

from collections.abc import Sequence

from rlm_train.attempts import AttemptRequest, AttemptRunner
from rlm_train.datasets.records import DatasetRecord
from rlm_train.evaluation.records import RecursiveEvaluationRecord
from rlm_train.evaluation.scoring import Scorer


class RecursiveEvaluationRunner:
    def __init__(self, attempt_runner: AttemptRunner, scorer: Scorer):
        self.attempt_runner = attempt_runner
        self.scorer = scorer

    def evaluate(
        self,
        records: Sequence[DatasetRecord],
        *,
        checkpoint_id: str,
        base_seed: int = 0,
    ) -> tuple[RecursiveEvaluationRecord, ...]:
        results = []
        for index, record in enumerate(records):
            attempt_result = self.attempt_runner.run(
                AttemptRequest(
                    task_id=record.record_id,
                    public_task=record.public_task,
                    private_reference=record.verifier_data,
                    mode="evaluation",
                )
            )
            final_answer = attempt_result.completion.response
            score, scoring = self.scorer.score(record, final_answer)
            results.append(
                RecursiveEvaluationRecord(
                    record_id=record.record_id,
                    # Evaluation schema version 1 retains ``rollout_id``.
                    rollout_id=attempt_result.attempt.rollout_id,
                    checkpoint_id=checkpoint_id,
                    sampling_seed=base_seed + index,
                    final_answer=final_answer,
                    score=score,
                    scoring=scoring,
                )
            )
        return tuple(results)


__all__ = ["RecursiveEvaluationRunner"]
