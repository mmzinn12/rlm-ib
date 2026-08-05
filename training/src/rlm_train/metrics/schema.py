"""Stable hierarchical scalar observations."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)+$")
    value: float
    step: int = Field(ge=0)
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_finite(self) -> MetricObservation:
        if not math.isfinite(self.value):
            raise ValueError("metric values must be finite")
        return self


__all__ = ["MetricObservation"]
