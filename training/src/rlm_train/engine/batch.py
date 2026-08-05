"""Objective-driven runtime batch planning, separate from durable JSON."""

from __future__ import annotations

from dataclasses import dataclass

from rlm_train.objectives.protocol import ObjectiveBatch, ObjectiveCapabilities
from rlm_train.spec.feedback import AssessmentScope


@dataclass(frozen=True)
class BatchRequirements:
    rollout_count: int
    behavior_logprobs: bool
    rewards: bool
    feedback_scopes: frozenset[AssessmentScope]
    teacher_targets: bool
    hidden_states: bool
    anchor_model: bool


def plan_batch_requirements(
    capabilities: dict[str, ObjectiveCapabilities],
) -> BatchRequirements:
    """Union only enabled objective declarations; disabled objectives are absent."""
    if not capabilities:
        raise ValueError("batch planning requires at least one enabled objective")
    return BatchRequirements(
        rollout_count=max(value.required_rollouts for value in capabilities.values()),
        behavior_logprobs=any(value.behavior_logprobs for value in capabilities.values()),
        rewards=any(value.rewards for value in capabilities.values()),
        feedback_scopes=frozenset(
            value.feedback_scope
            for value in capabilities.values()
            if value.feedback_scope is not None
        ),
        teacher_targets=any(value.teacher_targets for value in capabilities.values()),
        hidden_states=any(value.hidden_states for value in capabilities.values()),
        anchor_model=any(value.anchor_model for value in capabilities.values()),
    )


__all__ = ["BatchRequirements", "ObjectiveBatch", "plan_batch_requirements"]
