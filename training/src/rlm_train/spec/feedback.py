"""Feedback visibility and evidence-scope configuration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from rlm_train.spec.models import ImmutableSpec


class AssessmentScope(StrEnum):
    CAUSAL_LOCAL = "causal_local"
    RETROSPECTIVE_LOCAL = "retrospective_local"
    PRIVILEGED_DIAGNOSTIC = "privileged_diagnostic"


class FeedbackSpec(ImmutableSpec):
    default_scope: AssessmentScope = AssessmentScope.RETROSPECTIVE_LOCAL
    upstream_depth: int = Field(default=1, ge=0)
    downstream_depth: int = Field(default=1, ge=0)
    include_siblings: bool = False
    allow_privileged_hindsight_distillation: bool = False
    projector_version: str = Field(default="v1", min_length=1)


__all__ = ["AssessmentScope", "FeedbackSpec"]
