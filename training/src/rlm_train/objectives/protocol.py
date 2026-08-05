"""Standardized objective capabilities, batches, results, and attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rlm_train.feedback.schema import FeedbackBundle
from rlm_train.spec.feedback import AssessmentScope
from rlm_train.spec.objectives import TokenScope
from rlm_train.teachers.targets import TeacherTarget
from rlm_train.trajectory.schema import AnnotatedRollout, ObjectiveSelection


@dataclass(frozen=True)
class ObjectiveCapabilities:
    token_scope: TokenScope
    required_rollouts: int = 1
    behavior_logprobs: bool = False
    rewards: bool = False
    feedback_scope: AssessmentScope | None = None
    teacher_targets: bool = False
    hidden_states: bool = False
    anchor_model: bool = False


@dataclass(frozen=True)
class ObjectiveBatch:
    rollouts: tuple[AnnotatedRollout, ...]
    token_selections: dict[str, ObjectiveSelection]
    policy_scores: dict[str, Any] = field(default_factory=dict)
    behavior_policy_scores: dict[str, Any] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    advantages: dict[str, float] = field(default_factory=dict)
    feedback: FeedbackBundle | None = None
    teacher_targets: dict[str, TeacherTarget] = field(default_factory=dict)
    hidden_states: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossAttribution:
    rollout_id: str
    node_id: str
    generation_id: str
    token_start: int
    token_end: int
    value: float | None = None


@dataclass(frozen=True)
class ObjectiveResult:
    loss: Any
    active_token_count: int
    diagnostics: dict[str, float] = field(default_factory=dict)
    attributions: tuple[LossAttribution, ...] = ()


class Objective(Protocol):
    @property
    def capabilities(self) -> ObjectiveCapabilities: ...

    def compute(self, batch: ObjectiveBatch) -> ObjectiveResult: ...


__all__ = [
    "LossAttribution",
    "Objective",
    "ObjectiveBatch",
    "ObjectiveCapabilities",
    "ObjectiveResult",
]
