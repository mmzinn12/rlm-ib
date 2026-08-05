"""Resolve an immutable RunSpec into protocol-typed concrete components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlm_train.artifacts.provenance import RunProvenance
from rlm_train.spec import RunSpec

FACTORY_VERSION = "rlm-train-factory-v1"


@dataclass(frozen=True)
class ResolvedComponents:
    trainer: Any | None = None
    evaluator: Any | None = None
    rollout_engine: Any | None = None
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
        if component in self._builders:
            raise ValueError(f"component builder {component!r} is already registered")
        self._builders[component] = builder

    def resolve(
        self,
        spec: RunSpec,
        *,
        overrides: ResolvedComponents | None = None,
    ) -> ResolvedComponents:
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


__all__ = ["ComponentFactory", "FACTORY_VERSION", "ResolvedComponents"]
