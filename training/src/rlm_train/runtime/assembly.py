"""Spec-driven assembly composing the per-package build entry points into a runnable factory.

Every collaborator has a single ``build_*`` entry point in its own package; this module wires
them together into a ``CanonicalTrainer`` and a self-contained ``ComponentFactory``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from rlm_train.datasets.build import build_dataset
from rlm_train.datasets.protocol import Dataset
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.evaluation.scoring import Scorer
from rlm_train.rollouts.build import build_rollout_engine
from rlm_train.rollouts.rlm_engine import RLMRolloutEngine
from rlm_train.runtime.factory import ComponentFactory
from rlm_train.spec import RunSpec


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
    """Register the spec-constructible builders (dataset, rollout engine, evaluator) on a factory.

    Args:
        factory: Component factory to register builders on.
        policy: Shared trainable policy bound into the rollout engine and evaluator.
        scorer: Scorer used by the evaluator to grade held-out responses.
        held_out: Optional explicit held-out dataset; falls back to the first eval dataset.
        backend: RLM client backend forwarded to the rollout engine.
        environment_kwargs: Optional overrides forwarded to the RLM environment.
        checkpoint_id: Identifier recorded on evaluation records.
        predictions_filename: Name of the gradable predictions file the evaluator writes.
    """

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
    """Build a callable that yields a mixed-precision autocast context, or a no-op.

    Args:
        precision: Requested precision; ``"bf16"``/``"fp16"`` enable CUDA autocast when available.

    Returns:
        A zero-argument callable returning a context manager for use around the forward pass.
    """

    def factory() -> AbstractContextManager[Any]:
        if precision in {"bf16", "fp16"}:
            import torch

            if torch.cuda.is_available():
                dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    return factory


def build_canonical_trainer(
    run: RunSpec,
    *,
    policy: Any,
    judge: Any,
    resume_checkpoint: str | Path | None = None,
) -> Any:
    """Assemble a runnable CanonicalTrainer by composing the per-package build entry points.

    Args:
        run: Run specification supplying every sub-config the collaborators read.
        policy: Shared trainable policy that is scored, rolled out, and used as the teacher.
        judge: Judge used to produce scoped feedback for each rollout.

    Returns:
        A ``CanonicalTrainer`` wired with dataset, rollout engine, objectives, optimizer, scheduler,
        providers, teacher targets, and metric/artifact sinks.

    Raises:
        ValueError: If ``run.training_dataset`` is not set.
    """
    from rlm_train.artifacts.build import build_artifact_writer
    from rlm_train.artifacts.checkpoints import TransformersCheckpointWriter
    from rlm_train.engine.optimizer import build_optimizer
    from rlm_train.engine.providers import build_feedback_provider, build_policy_score_provider
    from rlm_train.engine.scheduler import build_training_scheduler
    from rlm_train.engine.trainer import CanonicalTrainer
    from rlm_train.metrics.build import build_metric_sink
    from rlm_train.objectives.build import build_objective_composer
    from rlm_train.teachers.build import build_teacher_target_provider, build_teachers

    if run.training_dataset is None:
        raise ValueError("training_dataset must be set to build a trainer")
    parameters = list(policy.trainable_parameters())
    optimizer = build_optimizer(parameters, run.runtime)
    scheduler = build_training_scheduler(
        optimizer, run.runtime, total_steps=run.runtime.max_optimizer_steps
    )
    checkpoint_writer = TransformersCheckpointWriter(
        run.artifacts.output_directory,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_interval=run.artifacts.checkpoint_interval,
        retain_checkpoints=run.artifacts.retain_checkpoints,
        save_final_checkpoint=run.artifacts.save_final_checkpoint,
    )
    initial_state = None
    if resume_checkpoint is not None:
        initial_state = checkpoint_writer.restore_training_state(
            resume_checkpoint,
            run_spec_fingerprint=run.fingerprint,
        )
    return CanonicalTrainer(
        spec=run,
        dataset=build_dataset(run.training_dataset),
        rollout_engine=build_rollout_engine(run, policy=policy),
        objectives=build_objective_composer(run.objectives),
        optimizer=optimizer,
        scheduler=scheduler,
        policy_scores=build_policy_score_provider(policy),
        policy_owner=run.student.resolved_policy_owner,
        policy_parameters=parameters,
        feedback=build_feedback_provider(judge),
        teacher_targets=build_teacher_target_provider(
            run.teacher, policy=policy, top_k=run.objectives.sdpo.top_k
        ),
        teachers=build_teachers(run.teacher, policy=policy),
        artifact_writer=build_artifact_writer(
            run.artifacts.output_directory, mode=run.artifacts.rollout_json
        ),
        metric_sink=(
            build_metric_sink(run.artifacts.output_directory)
            if run.artifacts.metrics_jsonl
            else None
        ),
        precision_context=precision_context_factory(run.runtime.precision),
        checkpoint_writer=checkpoint_writer,
        initial_state=initial_state,
        verbose=True,
    )


def assemble_default_factory(
    spec: RunSpec,
    *,
    scorer: Scorer | None = None,
    checkpoint_path: str | Path | None = None,
    resume_training: bool = False,
) -> ComponentFactory:
    """Load one shared policy and judge, then register all pipeline builders on a factory.

    Args:
        spec: Run specification whose ``student`` and ``judge`` drive the loaded policy and judge.
        scorer: Optional scorer; when provided, an evaluator builder is also registered.

    Returns:
        A ``ComponentFactory`` resolving policy, judge, dataset, rollout engine, trainer, and
        (optionally) evaluator.
    """
    from rlm_train.evaluation.build import build_evaluator
    from rlm_train.judge.providers import build_judge
    from rlm_train.models.build import build_policy

    factory = ComponentFactory()
    policy = build_policy(
        spec.student,
        runtime=spec.runtime,
        checkpoint_path=checkpoint_path,
    )
    judge = build_judge(spec.judge)

    factory.register("policy", lambda run: policy)
    factory.register("judge", lambda run: judge)
    factory.register("dataset", lambda run: build_dataset(run.training_dataset))
    factory.register("rollout_engine", lambda run: build_rollout_engine(run, policy=policy))
    factory.register(
        "trainer",
        lambda run: build_canonical_trainer(
            run,
            policy=policy,
            judge=judge,
            resume_checkpoint=checkpoint_path if resume_training else None,
        ),
    )
    if scorer is not None:
        factory.register(
            "evaluator",
            lambda run: build_evaluator(
                run,
                policy=policy,
                scorer=scorer,
                checkpoint_id=(
                    Path(checkpoint_path).name if checkpoint_path is not None else "base"
                ),
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
