"""Attach lazily resolved privileged evidence at the judge-only boundary.

Purpose:
    Let a future answer-key, assay, or reference-data source enrich judge requests
    without changing student prompts or persisted trajectory artifacts.
Implementation:
    ``PrivilegedContextTrajectoryJudge`` decorates any ``TrajectoryJudge``. It resolves
    context for the completed trajectory and passes a copied ``TaskContext`` only to the
    downstream judge.
Inputs:
    A provider, a downstream judge, a completed trajectory, and a public task context.
Outputs:
    The downstream judge's validated feedback.
Example:
    ``judge = PrivilegedContextTrajectoryJudge(base_judge, context_provider)``
"""

from __future__ import annotations

from rlm.core.trajectory import TrajectoryTree

from rlm_train.judge.base import TaskContext, TrajectoryJudge
from rlm_train.judge.context import PrivilegedContextProvider
from rlm_train.judge.schema import TrajectoryFeedback


class PrivilegedContextTrajectoryJudge:
    """Resolve optional privileged evidence immediately before judge evaluation."""

    def __init__(
        self,
        judge: TrajectoryJudge,
        provider: PrivilegedContextProvider,
    ) -> None:
        """Store the downstream judge and context provider."""
        self.judge = judge
        self.provider = provider

    async def evaluate(
        self,
        trajectory: TrajectoryTree,
        task: TaskContext,
    ) -> TrajectoryFeedback:
        """Resolve context and expose it only to the downstream judge call."""
        if task.privileged_context is not None:
            raise ValueError("privileged context is already attached to the task")
        context = await self.provider.get_context(
            task_id=task.task_id,
            trajectory=trajectory,
        )
        return await self.judge.evaluate(
            trajectory,
            task.with_privileged_context(context),
        )


__all__ = ["PrivilegedContextTrajectoryJudge"]
