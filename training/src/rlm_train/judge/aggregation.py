"""One-way aggregation from finalized local assessments to overall assessment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from rlm_train.feedback.schema import FeedbackVisibility, ScopedAssessment
from rlm_train.spec.feedback import AssessmentScope


def aggregate_overall_assessment(
    local_assessments: Sequence[ScopedAssessment],
    aggregate: Callable[[Sequence[dict[str, Any]]], dict[str, Any]],
    *,
    provider: str = "deterministic-aggregation",
    version: str = "v1",
) -> ScopedAssessment:
    if not local_assessments:
        raise ValueError("overall aggregation requires finalized local assessments")
    content = aggregate([assessment.content for assessment in local_assessments])
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


__all__ = ["aggregate_overall_assessment"]
