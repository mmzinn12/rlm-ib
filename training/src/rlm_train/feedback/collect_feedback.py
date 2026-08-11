"""Collect only the judge feedback required by enabled training methods."""

from __future__ import annotations

from rlm_train.attempts.attempt_records import AnnotatedAttempt
from rlm_train.feedback.aggregate_feedback import (
    FeedbackContentCombiner,
    create_overall_assessment,
)
from rlm_train.feedback.feedback_records import FeedbackBundle
from rlm_train.feedback.feedback_views import create_feedback_view
from rlm_train.judge.judge import FeedbackJudge
from rlm_train.settings.feedback import AssessmentScope


class FeedbackCollector:
    def __init__(
        self,
        judge: FeedbackJudge,
        *,
        combine_overall_content: FeedbackContentCombiner | None = None,
    ) -> None:
        self.judge = judge
        self.combine_overall_content = combine_overall_content

    def collect(
        self,
        attempts: tuple[AnnotatedAttempt, ...],
        scopes: frozenset[AssessmentScope],
    ) -> FeedbackBundle:
        assessments = []
        for attempt in attempts:
            for scope in sorted(scopes, key=lambda item: item.value):
                for edge in attempt.execution.edges:
                    view = create_feedback_view(
                        attempt,
                        scope=scope,
                        focal_edge_ids=(edge.edge_id,),
                    )
                    assessments.append(self.judge.assess(view))
        local_assessments = tuple(assessments)
        overall_assessment = (
            create_overall_assessment(local_assessments, self.combine_overall_content)
            if local_assessments and self.combine_overall_content is not None
            else None
        )
        return FeedbackBundle(
            local_assessments=local_assessments,
            overall_assessment=overall_assessment,
        )


__all__ = ["FeedbackCollector"]
