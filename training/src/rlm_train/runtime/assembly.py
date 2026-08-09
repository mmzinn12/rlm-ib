"""Spec-driven assembly of the components that are constructible without hand-injection.

This is the first assembly iteration: it wires the parts that derive cleanly from a RunSpec
(dataset, rollout engine, evaluator + readout). The heavy, environment-dependent components
(policy/model loading, objectives, teachers, optimizer, scheduler, trainer) are still injected
by the caller and will be folded into ``register_default_builders`` in a later iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from rlm_train.artifacts.rollout_json import RolloutJSONWriter
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


def precision_context_factory(precision: str) -> Callable[[], AbstractContextManager[Any]]:
    """Return a callable producing an autocast context for mixed precision, else a no-op."""

    def factory() -> AbstractContextManager[Any]:
        if precision in {"bf16", "fp16"}:
            import torch

            if torch.cuda.is_available():
                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    return factory


def build_canonical_trainer(run: RunSpec, *, policy: Any, judge: Any) -> Any:
    """Assemble a runnable CanonicalTrainer with concrete providers from a RunSpec."""
    import torch

    from rlm_train.engine.providers import (
        JudgeFeedbackProvider,
        SelfDistillationTeacherTargetProvider,
        TransformersPolicyScoreProvider,
    )
    from rlm_train.engine.trainer import CanonicalTrainer
    from rlm_train.metrics import JSONLMetricSink
    from rlm_train.objectives.build import build_objective_composer

    if run.training_dataset is None:
        raise ValueError("training_dataset must be set to build a trainer")
    parameters = list(policy.trainable_parameters())
    output = Path(run.artifacts.output_directory)
    return CanonicalTrainer(
        spec=run,
        dataset=build_dataset(run.training_dataset),
        rollout_engine=build_rollout_engine(run, policy=policy),
        objectives=build_objective_composer(run.objectives),
        optimizer=torch.optim.AdamW(parameters, lr=run.runtime.learning_rate),
        policy_scores=TransformersPolicyScoreProvider(policy),
        policy_owner=run.student.resolved_policy_owner,
        policy_parameters=parameters,
        feedback=JudgeFeedbackProvider(judge),
        teacher_targets=SelfDistillationTeacherTargetProvider(
            policy, top_k=run.objectives.sdpo.top_k
        ),
        artifact_writer=RolloutJSONWriter(output / "rollouts"),
        metric_sink=JSONLMetricSink(output / "metrics.jsonl"),
        precision_context=precision_context_factory(run.runtime.precision),
        verbose=True,
    )


def assemble_default_factory(spec: RunSpec, *, scorer: Scorer | None = None) -> ComponentFactory:
    """Load one shared policy and judge, then register all pipeline builders on a factory."""
    from rlm_train.judge.providers import build_judge
    from rlm_train.models.build import build_transformers_policy

    factory = ComponentFactory()
    policy = build_transformers_policy(spec.student, runtime=spec.runtime)
    judge = build_judge(spec.judge)

    factory.register("policy", lambda run: policy)
    factory.register("judge", lambda run: judge)
    factory.register("dataset", lambda run: build_dataset(run.training_dataset))
    factory.register("rollout_engine", lambda run: build_rollout_engine(run, policy=policy))
    factory.register(
        "trainer", lambda run: build_canonical_trainer(run, policy=policy, judge=judge)
    )
    if scorer is not None:
        factory.register(
            "evaluator",
            lambda run: RecursiveEvaluator(
                dataset=build_dataset(run.evaluation_datasets[0]),
                rollout_engine=build_rollout_engine(run, policy=policy),
                scorer=scorer,
                output_directory=run.artifacts.output_directory,
                checkpoint_id="latest",
                base_seed=run.evaluation.base_seed,
            ),
        )
    return factory


__all__ = [
    "assemble_default_factory",
    "build_canonical_trainer",
    "build_dataset",
    "build_rollout_engine",
    "precision_context_factory",
    "register_default_builders",
]