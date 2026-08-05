"""Whole-recursive-policy evaluation through the canonical rollout protocol."""

from __future__ import annotations

from collections.abc import Sequence

from rlm_train.datasets.records import DatasetRecord
from rlm_train.evaluation.records import RecursiveEvaluationRecord
from rlm_train.evaluation.scoring import Scorer
from rlm_train.rollouts.protocol import RolloutEngine, RolloutRequest


class RecursiveEvaluationRunner:
    def __init__(self, rollout_engine: RolloutEngine, scorer: Scorer):
        self.rollout_engine = rollout_engine
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
            rollout_result = self.rollout_engine.execute(
                RolloutRequest(
                    task_id=record.record_id,
                    public_task=record.public_task,
                    private_reference=record.verifier_data,
                    mode="evaluation",
                )
            )
            final_answer = rollout_result.completion.response
            score, scoring = self.scorer.score(record, final_answer)
            results.append(
                RecursiveEvaluationRecord(
                    record_id=record.record_id,
                    rollout_id=rollout_result.rollout.rollout_id,
                    checkpoint_id=checkpoint_id,
                    sampling_seed=base_seed + index,
                    final_answer=final_answer,
                    score=score,
                    scoring=scoring,
                )
            )
        return tuple(results)


__all__ = ["RecursiveEvaluationRunner"]
