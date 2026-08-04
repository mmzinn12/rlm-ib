"""Expose the provider-independent trajectory-judge API.

Purpose:
    Give callers one import surface for task context, judge protocols, and structured
    trajectory feedback.
Implementation:
    This package facade re-exports context isolation, judge protocols, structured
    execution, persistent caching, and strict feedback schemas.
Inputs:
    Python imports from rollout, evaluation, or training code.
Outputs:
    Public judge interfaces and validated feedback models.
Example:
    ``from rlm_train.judge import TaskContext, TrajectoryFeedback``
"""

from rlm_train.judge.base import TaskContext, TrajectoryJudge
from rlm_train.judge.cache import (
    FeedbackCache,
    MemoryFeedbackCache,
    SQLiteFeedbackCache,
    make_feedback_cache_key,
    make_trajectory_feedback_cache_key,
)
from rlm_train.judge.context import (
    PrivilegedContextDescriptor,
    PrivilegedContextProvider,
    PrivilegedJudgeContext,
)
from rlm_train.judge.privileged import PrivilegedContextTrajectoryJudge
from rlm_train.judge.schema import (
    DiagnosticQuestionTeacherFeedback,
    FactualQuestionTeacherFeedback,
    InformationValueFeedback,
    NodeFeedback,
    QuestionTeacherFeedback,
    TeacherFeedbackMode,
    TrajectoryFeedback,
)
from rlm_train.judge.structured import (
    JudgeExecutionMetrics,
    JudgeResponseError,
    StructuredJudgeClient,
    StructuredJudgeRequest,
    StructuredOutputTrajectoryJudge,
)

__all__ = [
    "FeedbackCache",
    "DiagnosticQuestionTeacherFeedback",
    "FactualQuestionTeacherFeedback",
    "InformationValueFeedback",
    "JudgeExecutionMetrics",
    "JudgeResponseError",
    "MemoryFeedbackCache",
    "NodeFeedback",
    "PrivilegedContextDescriptor",
    "PrivilegedContextProvider",
    "PrivilegedContextTrajectoryJudge",
    "PrivilegedJudgeContext",
    "QuestionTeacherFeedback",
    "TeacherFeedbackMode",
    "SQLiteFeedbackCache",
    "StructuredJudgeClient",
    "StructuredJudgeRequest",
    "StructuredOutputTrajectoryJudge",
    "TaskContext",
    "TrajectoryFeedback",
    "TrajectoryJudge",
    "make_feedback_cache_key",
    "make_trajectory_feedback_cache_key",
]
