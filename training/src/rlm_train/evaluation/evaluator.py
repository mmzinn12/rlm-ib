"""No-argument evaluator that runs the held-out set and writes gradable predictions."""

from __future__ import annotations

from pathlib import Path

from rlm_train.datasets.protocol import Dataset
from rlm_train.evaluation.predictions import PredictionsJSONLWriter
from rlm_train.evaluation.records import RecursiveEvaluationRecord
from rlm_train.evaluation.runner import RecursiveEvaluationRunner
from rlm_train.evaluation.scoring import Scorer
from rlm_train.rollouts.protocol import RolloutEngine


class RecursiveEvaluator:
    """Score the held-out dataset through the recursive policy and persist predictions."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        rollout_engine: RolloutEngine,
        scorer: Scorer,
        output_directory: str | Path,
        checkpoint_id: str,
        base_seed: int = 0,
        predictions_filename: str = "predictions.jsonl",
    ) -> None:
        self.dataset = dataset
        self.runner = RecursiveEvaluationRunner(rollout_engine, scorer)
        self.output_directory = Path(output_directory)
        self.checkpoint_id = checkpoint_id
        self.base_seed = base_seed
        self.predictions_filename = predictions_filename

    def evaluate(self) -> tuple[RecursiveEvaluationRecord, ...]:
        records = tuple(self.dataset.records())
        if not records:
            raise ValueError("evaluation dataset is empty")
        results = self.runner.evaluate(
            records,
            checkpoint_id=self.checkpoint_id,
            base_seed=self.base_seed,
        )
        questions = {record.record_id: record.public_task.get("prompt") for record in records}
        writer = PredictionsJSONLWriter(self.output_directory / self.predictions_filename)
        writer.write_all(results, questions=questions)
        return results


__all__ = ["RecursiveEvaluator"]
