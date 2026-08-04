"""Declare and validate the locked depth-1 tree-aware SDPO configuration.

Purpose:
    Encode the initial implementation decisions in one immutable configuration object.
Implementation:
    Pydantic literal fields lock reverse KL, EMA teaching, tail-aware top-k targets,
    same-tokenizer enforcement, exclusive masks, and depth 1. Component weights remain
    configurable but cannot all be zero.
Inputs:
    Optional top-k size, EMA rate, and non-negative component weights.
Outputs:
    An immutable, validated ``SDPOConfig``.
Example:
    ``config = SDPOConfig(top_k=100, ema_update_rate=0.05)``
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TeacherStrategy(StrEnum):
    """Select the lifecycle used for a run's policy teacher."""

    NONE = "none"
    FIXED = "fixed"
    EMA = "ema"


class ComponentWeights(BaseModel):
    """Configure non-negative weights for each normalized SDPO component.

    Attributes:
        route: Weight for recursion-routing decisions.
        call: Weight for call-construction decisions.
        node: Weight for child-node reasoning.
        aggregation: Weight for parent synthesis.
        final: Weight for final-answer decisions.
        missing_call: Weight for counterfactual missing-call feedback.

    Example:
        ``ComponentWeights(call=2.0, final=1.5)``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: float = Field(default=1.0, ge=0.0)
    call: float = Field(default=1.0, ge=0.0)
    node: float = Field(default=1.0, ge=0.0)
    aggregation: float = Field(default=1.0, ge=0.0)
    final: float = Field(default=1.0, ge=0.0)
    missing_call: float = Field(default=1.0, ge=0.0)


class SDPOConfig(BaseModel):
    """Hold the immutable settings for the initial tree-aware SDPO objective.

    Attributes:
        divergence: Fixed to reverse KL, ``D_KL(student || teacher)``.
        teacher: Fixed initial policy by default, or an EMA control teacher.
        top_k: Number of explicit teacher vocabulary entries per token.
        include_tail_bucket: Requires one aggregate probability for non-top-k tokens.
        ema_update_rate: Teacher update rate applied after optimizer steps.
        require_same_tokenizer: Requires teacher and student tokenizer fingerprints.
        mask_overlap: Fixed to exclusive component ownership.
        max_depth: Fixed to one child level for the initial implementation.
        allow_privileged_evidence: Disabled until leakage-safe handling is designed.
        component_weights: Non-negative weights applied after token normalization.

    Raises:
        pydantic.ValidationError: If a locked setting is changed or values are invalid.

    Example:
        ``config = SDPOConfig(component_weights=ComponentWeights(call=2.0))``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    divergence: Literal["reverse_kl"] = "reverse_kl"
    teacher: Literal[TeacherStrategy.FIXED, TeacherStrategy.EMA] = TeacherStrategy.FIXED
    top_k: int = Field(default=100, gt=0)
    include_tail_bucket: Literal[True] = True
    ema_update_rate: float | None = Field(default=None, gt=0.0, le=1.0)
    require_same_tokenizer: Literal[True] = True
    mask_overlap: Literal["exclusive"] = "exclusive"
    max_depth: Literal[1] = 1
    allow_privileged_evidence: Literal[False] = False
    component_weights: ComponentWeights = Field(default_factory=ComponentWeights)

    @model_validator(mode="after")
    def validate_active_objective(self) -> SDPOConfig:
        """Require at least one non-zero component weight.

        Returns:
            This validated configuration.

        Raises:
            ValueError: If every SDPO component is disabled.
        """
        if self.teacher is TeacherStrategy.EMA and self.ema_update_rate is None:
            raise ValueError("EMA teaching requires ema_update_rate")
        if self.teacher is TeacherStrategy.FIXED and self.ema_update_rate is not None:
            raise ValueError("fixed teaching does not accept ema_update_rate")
        weights = self.component_weights.model_dump().values()
        if not any(weight > 0 for weight in weights):
            raise ValueError("at least one SDPO component weight must be positive")
        return self


__all__ = ["ComponentWeights", "SDPOConfig", "TeacherStrategy"]
