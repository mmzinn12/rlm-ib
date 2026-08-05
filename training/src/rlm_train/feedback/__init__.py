"""Normalized feedback and privacy-preserving projections."""

from rlm_train.feedback.projection import project_feedback
from rlm_train.feedback.schema import (
    EnvironmentFeedback,
    FeedbackBundle,
    FeedbackProjection,
    FeedbackVisibility,
    ScopedAssessment,
)

__all__ = [
    "EnvironmentFeedback",
    "FeedbackBundle",
    "FeedbackProjection",
    "FeedbackVisibility",
    "ScopedAssessment",
    "project_feedback",
]
