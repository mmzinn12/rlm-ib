"""Evaluator construction from a RunSpec, a shared policy, and a scorer."""

from __future__ import annotations

from typing import Any

from rlm_train.datasets.build import build_dataset
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.evaluation.scoring import Scorer
from rlm_train.rollouts.build import build_rollout_engine
from rlm_train.spec import RunSpec


def build_evaluator(
    run: RunSpec,
    *,
    policy: Any,
    scorer: Scorer,
    backend: str = "openai",
    checkpoint_id: str = "latest",
    predictions_filename: str = "predictions.jsonl",
) -> RecursiveEvaluator:
    """Build a RecursiveEvaluator over the run's first held-out dataset.

    Args:
        run: Run specification; the first entry of ``evaluation_datasets`` is scored.
        policy: Shared trainable policy used to generate evaluation rollouts.
        scorer: Scorer that grades each final answer.
        backend: RLM client backend forwarded to the rollout engine.
        checkpoint_id: Identifier recorded on each evaluation record.
        predictions_filename: Name of the gradable predictions file written to the output dir.

    Returns:
        A ``RecursiveEvaluator`` ready to score the held-out set.

    Raises:
        ValueError: If ``run.evaluation_datasets`` is empty.
    """
    if not run.evaluation_datasets:
        raise ValueError("evaluation_datasets must be set to build an evaluator")
    return RecursiveEvaluator(
        dataset=build_dataset(run.evaluation_datasets[0]),
        rollout_engine=build_rollout_engine(run, policy=policy, backend=backend),
        scorer=scorer,
        output_directory=run.artifacts.output_directory,
        checkpoint_id=checkpoint_id,
        base_seed=run.evaluation.base_seed,
        predictions_filename=predictions_filename,
    )


__all__ = ["build_evaluator"]
