"""One-way combination of finalized local feedback into an overall assessment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from rlm_train.feedback.feedback_records import FeedbackVisibility, ScopedAssessment
from rlm_train.settings.feedback import AssessmentScope

FeedbackContentCombiner = Callable[[Sequence[dict[str, Any]]], dict[str, Any]]


def create_overall_assessment(
    local_assessments: Sequence[ScopedAssessment],
    combine_content: FeedbackContentCombiner,
    *,
    provider: str = "deterministic-aggregation",
    version: str = "v1",
) -> ScopedAssessment:
    """Combine completed local assessments without making the result trainable."""
    if not local_assessments:
        raise ValueError("overall feedback requires finalized local assessments")
    content = combine_content([assessment.content for assessment in local_assessments])
    view_identity = {
        "assessment_ids": [assessment.assessment_id for assessment in local_assessments],
        "view_fingerprints": [
            assessment.judge_view_fingerprint for assessment in local_assessments
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(view_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assessment_id = hashlib.sha256(f"overall:{fingerprint}".encode()).hexdigest()
    return ScopedAssessment(
        assessment_id=assessment_id,
        scope=AssessmentScope.PRIVILEGED_DIAGNOSTIC,
        evidence_node_ids=tuple(
            sorted({value for item in local_assessments for value in item.evidence_node_ids})
        ),
        evidence_event_ids=tuple(
            sorted({value for item in local_assessments for value in item.evidence_event_ids})
        ),
        judge_view_fingerprint=fingerprint,
        content=content,
        visibility=FeedbackVisibility.RESTRICTED,
        allowed_objectives=frozenset(),
        allowed_token_scopes=frozenset(),
        provider=provider,
        model_revision=version,
        prompt_version=version,
        cache_key=assessment_id,
    )


__all__ = ["FeedbackContentCombiner", "create_overall_assessment"]
