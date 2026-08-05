"""Canonical annotated rollout schema, projection, validation, and replay."""

from rlm_train.trajectory.projection import RolloutProjectionPolicy, project_rollout
from rlm_train.trajectory.replay import load_annotated_rollout, replay_annotated_events
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    AnnotationRecord,
    ContentKind,
    DecisionRole,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    FeedbackRecord,
    GenerationTokens,
    NodeRole,
    ObjectiveSelection,
    SemanticSpan,
    TaskPartition,
    Visibility,
)
from rlm_train.trajectory.validation import validate_annotated_rollout

__all__ = [
    "AnnotatedRollout",
    "AnnotationRecord",
    "ContentKind",
    "DecisionRole",
    "ExecutionEdge",
    "ExecutionNode",
    "ExecutionRecord",
    "FeedbackRecord",
    "GenerationTokens",
    "NodeRole",
    "ObjectiveSelection",
    "SemanticSpan",
    "TaskPartition",
    "Visibility",
    "load_annotated_rollout",
    "RolloutProjectionPolicy",
    "project_rollout",
    "replay_annotated_events",
    "validate_annotated_rollout",
]
