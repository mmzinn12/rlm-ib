"""Normalized environment and judge feedback with evidence provenance."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlm_train.spec.feedback import AssessmentScope
from rlm_train.spec.objectives import TokenScope


class ImmutableFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeedbackVisibility(StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    PRIVILEGED = "privileged"


class EnvironmentFeedback(ImmutableFeedback):
    feedback_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content: dict[str, Any]
    visibility: FeedbackVisibility = FeedbackVisibility.PUBLIC


class ScopedAssessment(ImmutableFeedback):
    assessment_id: str = Field(min_length=1)
    scope: AssessmentScope
    focal_node_ids: tuple[str, ...] = ()
    focal_edge_ids: tuple[str, ...] = ()
    evidence_node_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    judge_view_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: dict[str, Any]
    visibility: FeedbackVisibility
    future_public_events_visible: bool = False
    final_answer_visible: bool = False
    reference_answer_visible: bool = False
    allowed_objectives: frozenset[str] = frozenset()
    allowed_token_scopes: frozenset[TokenScope] = frozenset()
    provider: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    cache_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope_visibility(self) -> ScopedAssessment:
        if self.scope is AssessmentScope.CAUSAL_LOCAL:
            if (
                self.future_public_events_visible
                or self.final_answer_visible
                or self.reference_answer_visible
            ):
                raise ValueError("causal-local assessment cannot see future or privileged evidence")
        if self.scope is AssessmentScope.RETROSPECTIVE_LOCAL and self.reference_answer_visible:
            raise ValueError("retrospective-local assessment cannot see a private reference")
        if self.reference_answer_visible and self.visibility is not FeedbackVisibility.PRIVILEGED:
            raise ValueError("reference-visible assessment must be privileged")
        return self


class FeedbackProjection(ImmutableFeedback):
    projection_id: str = Field(min_length=1)
    assessment_ids: tuple[str, ...]
    objective: str = Field(min_length=1)
    token_scope: TokenScope
    content: dict[str, Any]
    visibility: FeedbackVisibility
    view_fingerprints: tuple[str, ...]
    projector_name: str = Field(min_length=1)
    projector_version: str = Field(min_length=1)


class FeedbackBundle(ImmutableFeedback):
    environment: tuple[EnvironmentFeedback, ...] = ()
    local_assessments: tuple[ScopedAssessment, ...] = ()
    projections: tuple[FeedbackProjection, ...] = ()
    overall_assessment: ScopedAssessment | None = None

    @model_validator(mode="after")
    def validate_unique_ids(self) -> FeedbackBundle:
        identifiers = [item.assessment_id for item in self.local_assessments]
        if self.overall_assessment is not None:
            identifiers.append(self.overall_assessment.assessment_id)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("feedback assessment IDs must be unique")
        return self


__all__ = [
    "EnvironmentFeedback",
    "FeedbackBundle",
    "FeedbackProjection",
    "FeedbackVisibility",
    "ScopedAssessment",
]
