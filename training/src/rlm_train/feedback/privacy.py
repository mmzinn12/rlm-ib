"""One-way feedback visibility enforcement."""

from __future__ import annotations

from rlm_train.feedback.schema import FeedbackVisibility

_VISIBILITY_RANK = {
    FeedbackVisibility.PUBLIC: 0,
    FeedbackVisibility.RESTRICTED: 1,
    FeedbackVisibility.PRIVILEGED: 2,
}


def ensure_projection_does_not_increase_visibility(
    source: FeedbackVisibility, projected: FeedbackVisibility
) -> None:
    if _VISIBILITY_RANK[projected] < _VISIBILITY_RANK[source]:
        raise ValueError("feedback projection cannot widen access to restricted evidence")


def require_teacher_safe(visibility: FeedbackVisibility, *, privileged_opt_in: bool) -> None:
    if visibility is FeedbackVisibility.PRIVILEGED and not privileged_opt_in:
        raise PermissionError("privileged feedback cannot enter teacher context without opt-in")


__all__ = ["ensure_projection_does_not_increase_visibility", "require_teacher_safe"]
