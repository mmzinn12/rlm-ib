"""Prepared exact-token inputs shared by training methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm_train.attempts.attempt_records import AnnotatedAttempt
from rlm_train.feedback.feedback_records import FeedbackBundle
from rlm_train.sdpo.feedback_predictions import FeedbackPredictions
from rlm_train.token_selection.selection import TokenSelection


@dataclass(frozen=True)
class StudentPredictionBatch:
    logits: dict[str, Any]
    behavior_logprobs: dict[str, Any] = field(default_factory=dict)
    hidden_states: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingBatch:
    attempts: tuple[AnnotatedAttempt, ...]
    token_selections: dict[str, TokenSelection]
    student_predictions: StudentPredictionBatch
    rewards: dict[str, float] = field(default_factory=dict)
    advantages: dict[str, float] = field(default_factory=dict)
    feedback: FeedbackBundle | None = None
    feedback_predictions: dict[str, FeedbackPredictions] = field(default_factory=dict)


@dataclass(frozen=True)
class LossAttribution:
    attempt_id: str
    node_id: str
    generation_id: str
    token_start: int
    token_end: int
    value: float | None = None


@dataclass(frozen=True)
class LossResult:
    loss: Any
    active_token_count: int
    diagnostics: dict[str, float] = field(default_factory=dict)
    attributions: tuple[LossAttribution, ...] = ()


__all__ = ["LossAttribution", "LossResult", "StudentPredictionBatch", "TrainingBatch"]
