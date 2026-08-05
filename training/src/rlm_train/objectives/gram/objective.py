"""Gram representation-drift objective."""

from __future__ import annotations

from collections.abc import Callable

from rlm_train.objectives.gram.math import gram_matrix_loss, multi_layer_gram_loss
from rlm_train.objectives.protocol import (
    ObjectiveBatch,
    ObjectiveCapabilities,
    ObjectiveResult,
)
from rlm_train.spec.objectives import GramSpec


class GramObjective:
    def __init__(
        self,
        spec: GramSpec,
        compute_loss: Callable[[ObjectiveBatch], ObjectiveResult],
    ) -> None:
        if not spec.enabled:
            raise ValueError("GramObjective requires an enabled specification")
        self.spec = spec
        self.compute_loss = compute_loss

    @property
    def capabilities(self) -> ObjectiveCapabilities:
        return ObjectiveCapabilities(
            token_scope=self.spec.token_scope,
            hidden_states=True,
            anchor_model=True,
        )

    def compute(self, batch: ObjectiveBatch) -> ObjectiveResult:
        if not batch.hidden_states:
            raise ValueError("Gram batch requires configured hidden-state captures")
        return self.compute_loss(batch)


__all__ = ["GramObjective", "gram_matrix_loss", "multi_layer_gram_loss"]
