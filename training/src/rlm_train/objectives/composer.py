"""The only component allowed to combine weighted objective results."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rlm_train.objectives.protocol import (
    Objective,
    ObjectiveBatch,
    ObjectiveCapabilities,
    ObjectiveResult,
)


@dataclass(frozen=True)
class ComposedObjectiveResult:
    loss: object
    results: dict[str, ObjectiveResult]
    weighted_losses: dict[str, object]


class ObjectiveComposer:
    def __init__(self, objectives: dict[str, tuple[float, Objective]]) -> None:
        if not objectives:
            raise ValueError("objective composer requires at least one enabled objective")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight, _ in objectives.values()):
            raise ValueError("enabled objective weights must be finite and positive")
        self.objectives = dict(objectives)

    @property
    def capabilities(self) -> dict[str, ObjectiveCapabilities]:
        return {name: objective.capabilities for name, (_, objective) in self.objectives.items()}

    def compute(self, batches: dict[str, ObjectiveBatch]) -> ComposedObjectiveResult:
        if set(batches) != set(self.objectives):
            raise ValueError("composer requires exactly one prepared batch per enabled objective")
        results: dict[str, ObjectiveResult] = {}
        weighted: dict[str, object] = {}
        total = None
        for name, (weight, objective) in self.objectives.items():
            result = objective.compute(batches[name])
            if result.active_token_count <= 0:
                raise ValueError(f"{name} objective selected no active tokens")
            loss = result.loss
            if hasattr(loss, "numel"):
                if loss.numel() != 1:
                    raise TypeError(f"{name} objective loss must be scalar")
                torch = __import__("torch")
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"{name} objective loss is non-finite")
            elif not math.isfinite(float(loss)):
                raise FloatingPointError(f"{name} objective loss is non-finite")
            value = result.loss * weight
            results[name] = result
            weighted[name] = value
            total = value if total is None else total + value
        assert total is not None
        return ComposedObjectiveResult(loss=total, results=results, weighted_losses=weighted)


__all__ = ["ComposedObjectiveResult", "ObjectiveComposer"]
