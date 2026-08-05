"""Masked SDPO objective and standardized capability declaration."""

from __future__ import annotations

from collections.abc import Callable

from rlm_train.objectives.protocol import (
    ObjectiveBatch,
    ObjectiveCapabilities,
    ObjectiveResult,
)
from rlm_train.spec.objectives import SDPOSpec


class SDPOObjective:
    """Keep algorithm-specific tensor construction cohesive behind one objective."""

    def __init__(
        self,
        spec: SDPOSpec,
        compute_loss: Callable[[ObjectiveBatch], ObjectiveResult],
    ) -> None:
        if not spec.enabled:
            raise ValueError("SDPOObjective requires an enabled specification")
        self.spec = spec
        self.compute_loss = compute_loss

    @property
    def capabilities(self) -> ObjectiveCapabilities:
        return ObjectiveCapabilities(
            token_scope=self.spec.token_scope,
            feedback_scope=self.spec.feedback_scope,
            teacher_targets=True,
        )

    def compute(self, batch: ObjectiveBatch) -> ObjectiveResult:
        if not batch.teacher_targets or batch.feedback is None:
            raise ValueError("SDPO batch requires teacher targets and scoped feedback")
        return self.compute_loss(batch)


__all__ = ["SDPOObjective"]
