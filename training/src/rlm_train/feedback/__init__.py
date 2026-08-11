"""Normalized feedback and privacy-preserving projections."""

from rlm_train.feedback.aggregate_feedback import create_overall_assessment
from rlm_train.feedback.collect_feedback import FeedbackCollector
from rlm_train.feedback.feedback_views import FeedbackView, create_feedback_view
from rlm_train.feedback.projection import project_feedback
from rlm_train.feedback.schema import (
    EnvironmentFeedback,
    FeedbackBundle,
    FeedbackProjection,
    FeedbackVisibility,
    ScopedAssessment,
)

__all__ = [
    "FeedbackCollector",
    "FeedbackView",
    "EnvironmentFeedback",
    "FeedbackBundle",
    "FeedbackProjection",
    "FeedbackVisibility",
    "ScopedAssessment",
    "create_feedback_view",
    "create_overall_assessment",
    "project_feedback",
]
