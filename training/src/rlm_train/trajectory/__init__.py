"""Expose trajectory recording, segmentation, and compilation APIs.

Purpose:
    Provide one import surface for the full trace-to-training-example workflow.
Implementation:
    This facade re-exports the recorder, deterministic segmenter, compiler, versioned
    artifact store, and offline replay types without adding runtime behavior.
Inputs:
    Python imports from rollout instrumentation, judges, or trainer adapters.
Outputs:
    Public trajectory workflow classes.
Example:
    ``from rlm_train.trajectory import TrajectoryRecorder, TrajectoryCompiler``
"""

from rlm_train.trajectory.artifacts import (
    TRAJECTORY_ARTIFACT_SCHEMA_VERSION,
    JSONLTrajectoryStore,
    TrajectoryArtifact,
    TrajectoryArtifactStore,
)
from rlm_train.trajectory.compiler import (
    NodeTrainingExample,
    QuestionTrainingExample,
    TrajectoryCompiler,
)
from rlm_train.trajectory.recorder import TrajectoryRecorder
from rlm_train.trajectory.replay import (
    OfflineTrajectoryReplay,
    ReplayCompilation,
    ReplayTokenizer,
    TokenizedContinuation,
    TokenizedNodeTrainingExample,
    TokenizedQuestionTrainingExample,
)
from rlm_train.trajectory.segmenter import RLMResponseSegmenter, RootSegmentation
from rlm_train.trajectory.validation import QuestionTraceMetrics, summarize_question_trace

__all__ = [
    "JSONLTrajectoryStore",
    "NodeTrainingExample",
    "OfflineTrajectoryReplay",
    "QuestionTraceMetrics",
    "QuestionTrainingExample",
    "RLMResponseSegmenter",
    "ReplayCompilation",
    "ReplayTokenizer",
    "RootSegmentation",
    "TRAJECTORY_ARTIFACT_SCHEMA_VERSION",
    "TokenizedContinuation",
    "TokenizedNodeTrainingExample",
    "TokenizedQuestionTrainingExample",
    "TrajectoryArtifact",
    "TrajectoryArtifactStore",
    "TrajectoryCompiler",
    "TrajectoryRecorder",
    "summarize_question_trace",
]
