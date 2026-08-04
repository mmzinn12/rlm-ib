"""Define provider-independent inputs and outputs for trajectory judges.

Purpose:
    Decouple trajectory evaluation from OpenAI, local inference, or any other judge
    implementation.
Implementation:
    ``TaskContext`` carries task evidence and ``TrajectoryJudge`` specifies one async
    evaluation method returning the strict schema from ``judge.schema``.
Inputs:
    A validated trajectory tree plus the task prompt, evidence snapshot, and metadata.
Outputs:
    A ``TrajectoryFeedback`` object keyed to trajectory node IDs.
Example:
    ``feedback = await judge.evaluate(tree, TaskContext("task-1", prompt))``
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from rlm.core.trajectory import TrajectoryTree

from rlm_train.judge.context import PrivilegedContextDescriptor, PrivilegedJudgeContext
from rlm_train.judge.schema import TrajectoryFeedback


@dataclass(frozen=True)
class TaskContext:
    """Bundle task information that a trajectory judge may inspect.

    Args:
        task_id: Stable task or example identifier.
        prompt: Original task prompt in its native representation.
        evidence_snapshot: Evidence available for evaluation at this point in time.
        metadata: Optional evaluator-specific context and version information.
        privileged_context: Optional judge-only evidence. This field is excluded from
            the public task payload and has a payload-free representation.

    Example:
        ``TaskContext(task_id="assay-1", prompt="Compare the controls")``
    """

    task_id: str
    prompt: Any
    evidence_snapshot: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    privileged_context: PrivilegedJudgeContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def public_payload(self) -> dict[str, Any]:
        """Return task data that is safe to persist or expose outside the judge.

        The privileged payload and its descriptor are intentionally absent. Artifact
        code persists the descriptor separately when provenance is required.
        """
        return {
            "task_id": self.task_id,
            "prompt": copy.deepcopy(self.prompt),
            "evidence_snapshot": copy.deepcopy(self.evidence_snapshot),
            "metadata": copy.deepcopy(self.metadata),
        }

    def judge_payload(self) -> dict[str, Any]:
        """Return public task data plus explicitly materialized judge-only evidence."""
        payload = self.public_payload()
        payload["privileged_context"] = (
            self.privileged_context.to_judge_payload()
            if self.privileged_context is not None
            else None
        )
        return payload

    def privileged_descriptor(self) -> PrivilegedContextDescriptor | None:
        """Return safe privileged provenance without revealing its content."""
        if self.privileged_context is None:
            return None
        return self.privileged_context.descriptor()

    def with_privileged_context(
        self,
        context: PrivilegedJudgeContext | None,
    ) -> TaskContext:
        """Return a new task context with the supplied judge-only evidence channel."""
        return replace(self, privileged_context=context)


class TrajectoryJudge(Protocol):
    """Specify the asynchronous contract implemented by a trajectory evaluator."""

    async def evaluate(self, trajectory: TrajectoryTree, task: TaskContext) -> TrajectoryFeedback:
        """Evaluate a complete rollout and return node-addressable feedback.

        Args:
            trajectory: Validated root/subcall trajectory to assess.
            task: Original task and evidence available to the evaluator.

        Returns:
            Structured feedback for final, node, and subcall information-value signals.
        """
        ...
