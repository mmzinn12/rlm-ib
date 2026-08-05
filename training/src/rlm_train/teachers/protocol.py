"""Teacher lifecycle and exact-target construction protocol."""

from __future__ import annotations

from typing import Protocol

from rlm_train.feedback.schema import FeedbackProjection
from rlm_train.models.protocol import SampledGeneration
from rlm_train.teachers.targets import TeacherTarget
from rlm_train.trajectory.schema import ObjectiveSelection


class Teacher(Protocol):
    def build_target(
        self,
        *,
        rollout_id: str,
        generation: SampledGeneration,
        selection: ObjectiveSelection,
        feedback: tuple[FeedbackProjection, ...],
    ) -> TeacherTarget: ...

    def after_optimizer_step(self) -> None: ...


__all__ = ["Teacher"]
