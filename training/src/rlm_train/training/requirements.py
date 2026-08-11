"""Declare and combine the data required by enabled training methods."""

from __future__ import annotations

from dataclasses import dataclass

from rlm_train.settings.feedback import AssessmentScope
from rlm_train.settings.token_selection import TokenScope


@dataclass(frozen=True)
class TrainingRequirements:
    included_text: TokenScope
    attempt_count: int = 1
    behavior_logprobs: bool = False
    rewards: bool = False
    feedback_scope: AssessmentScope | None = None
    feedback_predictions: bool = False
    hidden_states: bool = False
    anchor_model: bool = False


@dataclass(frozen=True)
class BatchRequirements:
    attempt_count: int
    behavior_logprobs: bool
    rewards: bool
    feedback_scopes: frozenset[AssessmentScope]
    feedback_predictions: bool
    hidden_states: bool
    anchor_model: bool


def combine_requirements(
    requirements: dict[str, TrainingRequirements],
) -> BatchRequirements:
    if not requirements:
        raise ValueError("training requires at least one enabled method")
    return BatchRequirements(
        attempt_count=max(value.attempt_count for value in requirements.values()),
        behavior_logprobs=any(value.behavior_logprobs for value in requirements.values()),
        rewards=any(value.rewards for value in requirements.values()),
        feedback_scopes=frozenset(
            value.feedback_scope for value in requirements.values() if value.feedback_scope
        ),
        feedback_predictions=any(value.feedback_predictions for value in requirements.values()),
        hidden_states=any(value.hidden_states for value in requirements.values()),
        anchor_model=any(value.anchor_model for value in requirements.values()),
    )


__all__ = ["BatchRequirements", "TrainingRequirements", "combine_requirements"]
