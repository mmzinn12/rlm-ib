"""Common interface implemented by feedback judges."""

from __future__ import annotations

from typing import Protocol

from rlm_train.feedback.feedback_records import ScopedAssessment
from rlm_train.feedback.feedback_views import FeedbackView


class FeedbackJudge(Protocol):
    def assess(self, view: FeedbackView) -> ScopedAssessment: ...


__all__ = ["FeedbackJudge"]
