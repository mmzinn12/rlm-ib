"""Canonical full-RLM rollout configuration."""

from __future__ import annotations

from pydantic import Field, model_validator

from rlm_train.spec.models import ImmutableSpec


class RolloutSpec(ImmutableSpec):
    engine: str = "rlm"
    environment: str = "local"
    max_depth: int = Field(default=2, ge=1)
    max_iterations: int = Field(default=20, gt=0)
    max_concurrent_subcalls: int = Field(default=4, gt=0)
    persistent: bool = False
    system_prompt: str | None = None
    sampling: dict[str, object] = Field(default_factory=dict)
    subcall_sampling: dict[str, object] | None = None

    @model_validator(mode="after")
    def require_canonical_engine(self) -> RolloutSpec:
        if self.engine != "rlm":
            raise ValueError("the canonical 'rlm' engine is the only supported rollout engine")
        return self


__all__ = ["RolloutSpec"]
