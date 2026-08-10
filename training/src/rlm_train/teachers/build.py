"""Teacher-target provider and lifecycle teacher construction from a TeacherSpec."""

from __future__ import annotations

from typing import Any

from rlm_train.engine.providers import SelfDistillationTeacherTargetProvider
from rlm_train.spec.models import TeacherSpec, TeacherStrategy


def build_teacher_target_provider(
    spec: TeacherSpec, *, policy: Any, top_k: int
) -> SelfDistillationTeacherTargetProvider:
    """Build the teacher-target provider for the configured teacher strategy.

    Args:
        spec: Teacher specification selecting the strategy.
        policy: Policy used as the (no-grad) teacher for self-distillation.
        top_k: Size of the retained teacher top-k support per selected token.

    Returns:
        A teacher-target provider producing top-k+tail targets.

    Raises:
        NotImplementedError: If the teacher strategy has no wired provider.
    """
    if spec.strategy is TeacherStrategy.CURRENT_POLICY:
        return SelfDistillationTeacherTargetProvider(policy, top_k=top_k)
    raise NotImplementedError(f"teacher strategy {spec.strategy.value!r} is not wired yet")


def build_teachers(spec: TeacherSpec, *, policy: Any) -> tuple[Any, ...]:
    """Return lifecycle teachers updated after each optimizer step (empty for self-distillation)."""
    if spec.strategy is TeacherStrategy.CURRENT_POLICY:
        return ()
    raise NotImplementedError(f"teacher strategy {spec.strategy.value!r} is not wired yet")


__all__ = ["build_teacher_target_provider", "build_teachers"]
