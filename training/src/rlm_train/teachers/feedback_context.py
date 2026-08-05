"""Safe feedback-conditioned teacher prompt context."""

from __future__ import annotations

import json
from collections.abc import Sequence

from rlm_train.feedback.privacy import require_teacher_safe
from rlm_train.feedback.schema import FeedbackProjection


def render_teacher_feedback_context(
    projections: Sequence[FeedbackProjection], *, privileged_opt_in: bool = False
) -> str:
    payload = []
    for projection in projections:
        require_teacher_safe(projection.visibility, privileged_opt_in=privileged_opt_in)
        payload.append(
            {
                "projection_id": projection.projection_id,
                "content": projection.content,
                "visibility": projection.visibility.value,
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = ["render_teacher_feedback_context"]
