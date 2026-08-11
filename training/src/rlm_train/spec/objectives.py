"""Independent objective and token-selection specifications."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from rlm_train.spec.feedback import AssessmentScope
from rlm_train.spec.models import ImmutableSpec


class TokenScope(StrEnum):
    NATURAL_LANGUAGE = "natural_language"
    HELPER_QUESTIONS = "helper_questions"
    SUBCALL_NATURAL_LANGUAGE = "subcall_natural_language"
    ALL_STUDENT_TOKENS = "all_student_tokens"


class ObjectiveSpec(ImmutableSpec):
    enabled: bool = False
    weight: float = Field(default=0.0, ge=0.0)
    token_scope: TokenScope = TokenScope.NATURAL_LANGUAGE

    @model_validator(mode="after")
    def validate_enabled_weight(self) -> ObjectiveSpec:
        if self.enabled and self.weight <= 0.0:
            raise ValueError("enabled objective requires a positive weight")
        if not self.enabled and self.weight != 0.0:
            raise ValueError("disabled objective must have zero weight")
        return self


class GRPOSpec(ObjectiveSpec):
    group_size: int = Field(default=4, gt=1)
    clip_epsilon: float = Field(default=0.2, gt=0.0)
    kl_coefficient: float = Field(default=0.0, ge=0.0)


class SDPOSpec(ObjectiveSpec):
    feedback_scope: AssessmentScope = AssessmentScope.RETROSPECTIVE_LOCAL
    divergence: Literal["reverse_kl"] = "reverse_kl"
    target_support: Literal["top_k_with_tail"] = "top_k_with_tail"
    top_k: int = Field(default=32, gt=0)


class GramSpec(ObjectiveSpec):
    layers: tuple[int, ...] = ()
    normalize: bool = True


class ObjectivesSpec(ImmutableSpec):
    grpo: GRPOSpec = Field(default_factory=GRPOSpec)
    sdpo: SDPOSpec = Field(default_factory=SDPOSpec)
    gram: GramSpec = Field(default_factory=GramSpec)

    @property
    def enabled(self) -> tuple[tuple[str, ObjectiveSpec], ...]:
        values = (("grpo", self.grpo), ("sdpo", self.sdpo), ("gram", self.gram))
        return tuple((name, spec) for name, spec in values if spec.enabled)


__all__ = [
    "GRPOSpec",
    "GramSpec",
    "ObjectiveSpec",
    "ObjectivesSpec",
    "SDPOSpec",
    "TokenScope",
]
