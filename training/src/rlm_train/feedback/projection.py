"""Pure objective-authorized judge-feedback projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from rlm_train.feedback.privacy import ensure_projection_does_not_increase_visibility
from rlm_train.feedback.schema import FeedbackProjection, FeedbackVisibility, ScopedAssessment
from rlm_train.spec.objectives import TokenScope


def project_feedback(
    assessments: Iterable[ScopedAssessment],
    *,
    objective: str,
    token_scope: TokenScope,
    projector_name: str,
    projector_version: str,
    visibility: FeedbackVisibility,
    fields: tuple[str, ...] | None = None,
) -> FeedbackProjection:
    values = tuple(assessments)
    if not values:
        raise ValueError("feedback projection requires at least one assessment")
    content: dict[str, Any] = {}
    for assessment in values:
        if objective not in assessment.allowed_objectives:
            raise PermissionError(
                f"assessment {assessment.assessment_id!r} is not authorized for {objective!r}"
            )
        if token_scope not in assessment.allowed_token_scopes:
            raise PermissionError(
                f"assessment {assessment.assessment_id!r} is not authorized for {token_scope.value!r}"
            )
        ensure_projection_does_not_increase_visibility(assessment.visibility, visibility)
        selected = assessment.content
        if fields is not None:
            selected = {field: selected[field] for field in fields if field in selected}
        content[assessment.assessment_id] = selected
    identity = {
        "assessments": [item.assessment_id for item in values],
        "objective": objective,
        "token_scope": token_scope.value,
        "projector": [projector_name, projector_version],
    }
    projection_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FeedbackProjection(
        projection_id=projection_id,
        assessment_ids=tuple(item.assessment_id for item in values),
        objective=objective,
        token_scope=token_scope,
        content=content,
        visibility=visibility,
        view_fingerprints=tuple(item.judge_view_fingerprint for item in values),
        projector_name=projector_name,
        projector_version=projector_version,
    )


__all__ = ["project_feedback"]
