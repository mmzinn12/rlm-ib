"""Resolve an immutable RunSpec into protocol-typed concrete components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlm_train.artifacts.provenance import RunProvenance
from rlm_train.attempts import AttemptRunner
from rlm_train.datasets.protocol import Dataset
from rlm_train.evaluation.evaluator import RecursiveEvaluator
from rlm_train.evaluation.scoring import Scorer
from rlm_train.judge.cache import JudgeCache
from rlm_train.judge.create_judge import create_judge
from rlm_train.judge.judge import FeedbackJudge
from rlm_train.spec import RunSpec

FACTORY_VERSION = "rlm-train-factory-v1"


@dataclass(frozen=True)
class ResolvedComponents:
    trainer: Any | None = None
    evaluator: Any | None = None
    attempt_runner: Any | None = None
    policy: Any | None = None
    dataset: Any | None = None
    objectives: Any | None = None
    judge: Any | None = None
    teacher: Any | None = None
    metrics: Any | None = None
    artifacts: Any | None = None
    identities: dict[str, Any] | None = None


class ComponentFactory:
    """Registry-based construction with direct injection for tests and research."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[RunSpec], Any]] = {}

    def register(self, component: str, builder: Callable[[RunSpec], Any]) -> None:
        """Register a builder for a named component; each component may be registered only once."""
        if component in self._builders:
            raise ValueError(f"component builder {component!r} is already registered")
        self._builders[component] = builder

    def resolve(
        self,
        spec: RunSpec,
        *,
        overrides: ResolvedComponents | None = None,
    ) -> ResolvedComponents:
        """Build each component from its registered builder unless injected via ``overrides``."""
        injected = overrides or ResolvedComponents()
        values: dict[str, Any] = {}
        for name in ResolvedComponents.__dataclass_fields__:
            current = getattr(injected, name)
            values[name] = current
            if current is None and name in self._builders:
                values[name] = self._builders[name](spec)
        return ResolvedComponents(**values)

    def provenance(self, spec: RunSpec, components: ResolvedComponents) -> RunProvenance:
        identities = dict(components.identities or {})
        return RunProvenance(
            run_spec_fingerprint=spec.fingerprint,
            resolved_spec=spec.resolved_dict(),
            components=identities,
            factory_version=FACTORY_VERSION,
        )


def register_judge_builder(
    factory: ComponentFactory,
    *,
    client: Any | None = None,
    cache: JudgeCache | None = None,
) -> None:
    """Register RunSpec-driven judge construction on the runtime factory."""

    def builder(run: RunSpec) -> FeedbackJudge:
        return create_judge(run.judge, client=client, cache=cache)

    factory.register("judge", builder)


def register_evaluator_builder(
    factory: ComponentFactory,
    *,
    dataset: Dataset,
    attempt_runner: AttemptRunner,
    scorer: Scorer,
    checkpoint_id: str,
    base_seed: int = 0,
) -> None:
    """Register RunSpec-driven evaluator construction on the runtime factory."""

    def builder(run: RunSpec) -> RecursiveEvaluator:
        return RecursiveEvaluator(
            dataset=dataset,
            attempt_runner=attempt_runner,
            scorer=scorer,
            output_directory=run.artifacts.output_directory,
            checkpoint_id=checkpoint_id,
            base_seed=base_seed,
        )

    factory.register("evaluator", builder)


__all__ = [
    "ComponentFactory",
    "FACTORY_VERSION",
    "ResolvedComponents",
    "register_evaluator_builder",
    "register_judge_builder",
]
