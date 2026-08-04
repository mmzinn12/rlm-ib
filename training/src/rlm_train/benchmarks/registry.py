"""Build benchmark adapters from explicit configuration without source edits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlm_train.benchmarks.jsonl import JSONLBenchmark
from rlm_train.benchmarks.types import Benchmark

BenchmarkFactory = Callable[[dict[str, Any]], Benchmark]


class BenchmarkRegistry:
    """Map stable adapter names to factories and reject accidental replacement."""

    def __init__(self) -> None:
        self._factories: dict[str, BenchmarkFactory] = {}

    def register(self, name: str, factory: BenchmarkFactory) -> None:
        """Register one non-blank unique adapter name."""
        if not name.strip():
            raise ValueError("benchmark adapter name must not be blank")
        if name in self._factories:
            raise ValueError(f"benchmark adapter {name!r} is already registered")
        self._factories[name] = factory

    def create(self, adapter: str, configuration: dict[str, Any]) -> Benchmark:
        """Create an adapter from a copied configuration mapping."""
        try:
            factory = self._factories[adapter]
        except KeyError as exc:
            raise ValueError(f"unknown benchmark adapter {adapter!r}") from exc
        return factory(dict(configuration))

    @property
    def adapter_names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._factories))


def default_benchmark_registry() -> BenchmarkRegistry:
    """Return a registry containing only download-free generic adapters."""
    registry = BenchmarkRegistry()
    registry.register("jsonl", lambda config: JSONLBenchmark(**config))
    return registry


__all__ = ["BenchmarkFactory", "BenchmarkRegistry", "default_benchmark_registry"]
