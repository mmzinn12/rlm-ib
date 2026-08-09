"""Spec-driven assembly of the components that are constructible without hand-injection.

This is the first assembly iteration: it wires the parts that derive cleanly from a RunSpec
(dataset, rollout engine, evaluator + readout). The heavy, environment-dependent components
(policy/model loading, objectives, teachers, optimizer, scheduler, trainer) are still injected
by the caller and will be folded into ``register_default_builders`` in a later iteration.
"""

from __future__ import annotations

from typing import Any

from rlm_train.datasets.adapters.jsonl import JSONLDataset
from rlm_train.datasets.protocol import Dataset
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.evaluation.scoring import Scorer
from rlm_train.rollouts.rlm_engine import RLMRolloutEngine
from rlm_train.runtime.factory import ComponentFactory
from rlm_train.spec import RunSpec
from rlm_train.spec.run import DatasetRefSpec


def build_dataset(ref: DatasetRefSpec) -> Dataset:
    if ref.adapter != "jsonl":
        raise ValueError(f"unsupported dataset adapter {ref.adapter!r}; only 'jsonl' is wired")
    return JSONLDataset(ref.source)


def build_rollout_engine(
    run: RunSpec,
    *,
    policy: Any,
    backend: str = "openai",
    environment_kwargs: dict[str, Any] | None = None,
) -> RLMRolloutEngine:
    return RLMRolloutEngine(
        policy=policy,
        policy_owner=run.student.resolved_policy_owner,
        spec=run.rollout,
        backend=backend,
        environment_kwargs=environment_kwargs,
    )


def register_default_builders(
    factory: ComponentFactory,
    *,
    policy: Any,
    scorer: Scorer,
    held_out: Dataset | None = None,
    backend: str = "openai",
    environment_kwargs: dict[str, Any] | None = None,
    checkpoint_id: str = "latest",
    predictions_filename: str = "predictions.jsonl",
) -> None:
    """Register the spec-constructible builders: dataset, rollout engine, and evaluator."""

    def dataset_builder(run: RunSpec) -> Dataset:
        if run.training_dataset is None:
            raise ValueError("training_dataset must be set to build a dataset")
        return build_dataset(run.training_dataset)

    def rollout_engine_builder(run: RunSpec) -> RLMRolloutEngine:
        return build_rollout_engine(
            run,
            policy=policy,
            backend=backend,
            environment_kwargs=environment_kwargs,
        )

    def evaluator_builder(run: RunSpec) -> RecursiveEvaluator:
        dataset = held_out
        if dataset is None:
            if not run.evaluation_datasets:
                raise ValueError("evaluation_datasets must be set to build an evaluator")
            dataset = build_dataset(run.evaluation_datasets[0])
        return RecursiveEvaluator(
            dataset=dataset,
            rollout_engine=build_rollout_engine(
                run,
                policy=policy,
                backend=backend,
                environment_kwargs=environment_kwargs,
            ),
            scorer=scorer,
            output_directory=run.artifacts.output_directory,
            checkpoint_id=checkpoint_id,
            base_seed=run.evaluation.base_seed,
            predictions_filename=predictions_filename,
        )

    factory.register("dataset", dataset_builder)
    factory.register("rollout_engine", rollout_engine_builder)
    factory.register("evaluator", evaluator_builder)


__all__ = ["build_dataset", "build_rollout_engine", "register_default_builders"]
