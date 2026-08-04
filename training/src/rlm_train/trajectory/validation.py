"""Summarize validation coverage for individually addressable question targets.

Purpose:
    Measure whether statically extracted question items were bound to runtime children
    and covered by judge feedback before question-level SDPO.
Implementation:
    An immutable metrics dataclass and a tree walker aggregate counts from broad call
    metadata and narrow ``CallItemSpan`` bindings without importing a judge or trainer.
Inputs:
    A validated ``TrajectoryTree`` and an optional number of question feedback records.
Outputs:
    ``QuestionTraceMetrics`` values or a tracker-ready dictionary.
Example:
    ``metrics = summarize_question_trace(tree, question_feedback_count=4)``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rlm.core.trajectory import DecisionKind, TrajectoryTree


@dataclass(frozen=True)
class QuestionTraceMetrics:
    """Summarize extraction, runtime binding, and judge coverage.

    Attributes:
        question_item_count: Total literal or explicitly unaddressable question items.
        bound_question_item_count: Addressable items bound to actual child nodes.
        unaddressable_question_item_count: Items that cannot receive an accurate mask.
        question_feedback_count: Edge-local judge feedback records available downstream.

    Example:
        ``metrics = QuestionTraceMetrics(10, 8, 2, 8)``
    """

    question_item_count: int
    bound_question_item_count: int
    unaddressable_question_item_count: int
    question_feedback_count: int = 0

    @property
    def addressable_question_item_count(self) -> int:
        """Return statically addressable items, whether or not they executed."""
        return self.question_item_count - self.unaddressable_question_item_count

    def to_dict(self) -> dict[str, int]:
        """Return metric names and integer values for tracker integrations.

        Returns:
            A new dictionary containing all four stored counts.
        """
        return asdict(self)


def summarize_question_trace(
    trajectory: TrajectoryTree, *, question_feedback_count: int = 0
) -> QuestionTraceMetrics:
    """Compute question-target coverage without judge or trainer dependencies.

    Args:
        trajectory: Tree whose call metadata and item spans should be counted.
        question_feedback_count: Optional non-negative downstream feedback count.

    Returns:
        Aggregate extraction, binding, unaddressable, and feedback counts.

    Raises:
        ValueError: If ``question_feedback_count`` is negative.

    Example:
        ``metrics = summarize_question_trace(tree, question_feedback_count=len(feedback.subcalls))``
    """
    if question_feedback_count < 0:
        raise ValueError("question feedback count must be non-negative")
    question_item_count = 0
    unaddressable_count = 0
    bound_count = 0
    for node in trajectory.nodes:
        node_question_count = 0
        node_unaddressable_count = 0
        for span in node.spans:
            if span.kind is not DecisionKind.CALL:
                continue
            node_question_count += int(span.metadata.get("question_item_count") or 0)
            node_unaddressable_count += int(
                span.metadata.get("unaddressable_question_item_count") or 0
            )
        question_item_count += max(
            node_question_count,
            len(node.call_item_spans) + node_unaddressable_count,
        )
        unaddressable_count += node_unaddressable_count
        bound_count += sum(item.child_node_id is not None for item in node.call_item_spans)
    return QuestionTraceMetrics(
        question_item_count=question_item_count,
        bound_question_item_count=bound_count,
        unaddressable_question_item_count=unaddressable_count,
        question_feedback_count=question_feedback_count,
    )


__all__ = ["QuestionTraceMetrics", "summarize_question_trace"]
