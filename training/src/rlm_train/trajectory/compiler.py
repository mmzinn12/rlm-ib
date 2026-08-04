"""Compile traced nodes and structured feedback into node-level training examples.

Purpose:
    Bridge rollout instrumentation and the tokenizer/trainer boundary while preserving
    node-local contexts, continuations, decision spans, and feedback.
Implementation:
    The compiler validates feedback edges, attaches subcall feedback to its parent,
    attaches final feedback to final root nodes, and reclassifies routing spans when a
    counterfactual missing call must be taught.
Inputs:
    A validated ``TrajectoryTree`` and matching ``TrajectoryFeedback``.
Outputs:
    ``NodeTrainingExample`` objects ready for tokenization and teacher scoring.
Example:
    ``examples = TrajectoryCompiler().compile(tree, feedback)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from rlm.core.trajectory import (
    CallItemSpan,
    DecisionKind,
    DecisionSpan,
    InvocationKind,
    TrajectoryTree,
)

from rlm_train.judge.schema import (
    QuestionTeacherFeedback,
    TeacherFeedbackMode,
    TrajectoryFeedback,
)


@dataclass(frozen=True)
class NodeTrainingExample:
    """Carry one traced node and its localized feedback to the trainer boundary.

    Args:
        trajectory_id: Stable rollout identifier.
        node_id: Stable invocation identifier.
        student_context: Feedback-free context visible to the student.
        continuation: Sampled policy response to tokenize and score.
        spans: Character-level component spans over ``continuation``.
        feedback: Judge feedback relevant to this node or its outgoing subcalls.
        policy_version: Optional version of the policy that sampled the continuation.

    Example:
        ``example = TrajectoryCompiler().compile(tree, feedback)[0]``
    """

    trajectory_id: str
    node_id: str
    student_context: Any
    continuation: str
    spans: list[DecisionSpan]
    feedback: dict[str, Any]
    policy_version: int | None


@dataclass(frozen=True)
class QuestionTrainingExample:
    """Carry exactly one question edge and its restricted teacher feedback.

    Attributes:
        trajectory_id: Stable rollout identifier.
        parent_node_id: Node whose continuation contains the question.
        child_node_id: Child invocation that answered the question.
        call_order: Zero-based order of the helper call in the parent continuation.
        batch_index: Position inside a batched call, or ``None`` for a scalar call.
        student_context: Feedback-free context visible to the student policy.
        student_continuation: Existing parent response whose tokens will be rescored.
        question_span: Exact response-relative span of this question expression.
        feedback: Edge-local teacher view with outcome and sibling data removed.

    Example:
        ``question = TrajectoryCompiler().compile_questions(tree, feedback)[0]``
    """

    trajectory_id: str
    parent_node_id: str
    child_node_id: str
    call_order: int
    batch_index: int | None
    student_context: Any
    student_continuation: str
    question_span: CallItemSpan
    feedback: QuestionTeacherFeedback


class TrajectoryCompiler:
    """Compile trajectory feedback without importing a tokenizer or trainer.

    Tokenization intentionally happens later so the same traced character spans can be
    mapped using the exact tokenizer instance shared by teacher and student.
    """

    def __init__(
        self,
        *,
        feedback_mode: TeacherFeedbackMode = TeacherFeedbackMode.DIAGNOSTIC,
        projector_version: str = "v1",
    ) -> None:
        """Lock one deterministic feedback projection for this compiler instance."""
        if not projector_version.strip():
            raise ValueError("projector_version must not be blank")
        self.feedback_mode = TeacherFeedbackMode(feedback_mode)
        self.projector_version = projector_version

    @property
    def projector_provenance(self) -> dict[str, str]:
        """Return the projection identity recorded with compiled trajectories."""
        return {
            "name": "edge_local_question_feedback",
            "version": self.projector_version,
            "mode": self.feedback_mode.value,
        }

    def compile(
        self, trajectory: TrajectoryTree, feedback: TrajectoryFeedback
    ) -> list[NodeTrainingExample]:
        """Validate and convert a trajectory into feedback-bearing node examples.

        Args:
            trajectory: Complete traced rollout containing stable node IDs and spans.
            feedback: Judge output referencing nodes and subcall edges in ``trajectory``.

        Returns:
            One example for each node with node, subcall, or final-answer feedback.
            Nodes without feedback are omitted.

        Raises:
            ValueError: If feedback references unknown nodes, assigns information value
                to a non-subcall, or disagrees with a traced parent-child edge.

        Example:
            ``examples = compiler.compile(trajectory=tree, feedback=feedback)``
        """
        feedback.validate_node_ids({node.node_id for node in trajectory.nodes})
        nodes_by_id = {node.node_id: node for node in trajectory.nodes}
        for item in feedback.subcalls:
            child = nodes_by_id[item.child_node_id]
            if child.kind is not InvocationKind.SUBCALL:
                raise ValueError("information-value feedback must reference a subcall child")
            if child.parent_id != item.parent_node_id:
                raise ValueError(
                    "information-value feedback parent must match the traced child edge"
                )
        feedback_by_node = {item.node_id: item for item in feedback.nodes}
        subcalls_by_parent: dict[str, list[dict[str, Any]]] = {}
        for item in feedback.subcalls:
            subcalls_by_parent.setdefault(item.parent_node_id, []).append(item.model_dump())
        final_node_ids = {
            node.node_id
            for node in trajectory.nodes
            if any(span.kind is DecisionKind.FINAL for span in node.spans)
        }
        if feedback.final_answer_feedback is not None and not final_node_ids:
            root_nodes = [node for node in trajectory.nodes if node.kind is InvocationKind.ROOT]
            if root_nodes:
                final_node_ids.add(root_nodes[-1].node_id)

        examples: list[NodeTrainingExample] = []
        for node in trajectory.nodes:
            node_feedback = feedback_by_node.get(node.node_id)
            payload = node_feedback.model_dump() if node_feedback is not None else {}
            if node.node_id in subcalls_by_parent:
                payload["subcalls"] = subcalls_by_parent[node.node_id]
            if node.node_id in final_node_ids and feedback.final_answer_feedback is not None:
                payload["final_answer_feedback"] = feedback.final_answer_feedback.model_dump()
            if not payload:
                continue
            spans = list(node.spans)
            if (
                node_feedback is not None
                and node_feedback.routing_feedback is not None
                and node_feedback.routing_feedback.missing_calls
            ):
                spans = [
                    DecisionSpan(
                        kind=DecisionKind.MISSING_CALL,
                        start=span.start,
                        end=span.end,
                        related_node_id=span.related_node_id,
                        metadata=dict(span.metadata),
                    )
                    if span.kind is DecisionKind.ROUTE
                    else span
                    for span in spans
                ]
            examples.append(
                NodeTrainingExample(
                    trajectory_id=trajectory.trajectory_id,
                    node_id=node.node_id,
                    student_context=node.context,
                    continuation=node.response,
                    spans=spans,
                    feedback=payload,
                    policy_version=node.policy_version,
                )
            )
        return examples

    def compile_questions(
        self,
        trajectory: TrajectoryTree,
        feedback: TrajectoryFeedback,
        *,
        on_unaddressable: Literal["error", "skip"] = "error",
    ) -> list[QuestionTrainingExample]:
        """Compile one leakage-restricted example per addressed subcall question.

        Args:
            trajectory: Traced rollout with runtime-bound ``CallItemSpan`` objects.
            feedback: Judge records containing edge-local information value.
            on_unaddressable: Fail when a judged edge has no exact item span, or skip it
                explicitly for experiments that permit dynamic question construction.

        Returns:
            Question examples ordered by parent storage order, call order, and batch
            index. No example can contain feedback from another child.

        Raises:
            ValueError: If the addressability policy is invalid, trajectory or feedback
                references are invalid, a bound edge disagrees with the trace, a child
                is bound more than once, or an addressed span is required but absent.

        Example:
            ``questions = compiler.compile_questions(tree, feedback, on_unaddressable="skip")``
        """
        if on_unaddressable not in {"error", "skip"}:
            raise ValueError("on_unaddressable must be 'error' or 'skip'")
        trajectory.validate()
        feedback.validate_node_ids({node.node_id for node in trajectory.nodes})
        nodes_by_id = {node.node_id: node for node in trajectory.nodes}
        feedback_by_child = {item.child_node_id: item for item in feedback.subcalls}
        examples: list[QuestionTrainingExample] = []
        addressed_child_ids: set[str] = set()
        for parent in trajectory.nodes:
            ordered_spans = sorted(
                parent.call_item_spans,
                key=lambda span: (
                    span.call_order,
                    -1 if span.batch_index is None else span.batch_index,
                ),
            )
            for span in ordered_spans:
                if span.child_node_id is None:
                    continue
                item_feedback = feedback_by_child.get(span.child_node_id)
                if item_feedback is None:
                    continue
                child = nodes_by_id[span.child_node_id]
                if child.parent_id != parent.node_id:
                    raise ValueError("bound question child must belong to the span parent")
                if item_feedback.parent_node_id != parent.node_id:
                    raise ValueError("question feedback parent must match the bound item parent")
                if span.child_node_id in addressed_child_ids:
                    raise ValueError("a child may be bound to only one question item")
                addressed_child_ids.add(span.child_node_id)
                examples.append(
                    QuestionTrainingExample(
                        trajectory_id=trajectory.trajectory_id,
                        parent_node_id=parent.node_id,
                        child_node_id=span.child_node_id,
                        call_order=span.call_order,
                        batch_index=span.batch_index,
                        student_context=parent.context,
                        student_continuation=parent.response,
                        question_span=span,
                        feedback=item_feedback.to_teacher_view(
                            self.feedback_mode,
                            projector_version=self.projector_version,
                        ),
                    )
                )
        unaddressed = set(feedback_by_child) - addressed_child_ids
        if unaddressed and on_unaddressable == "error":
            raise ValueError(
                "question feedback has no uniquely bound call-item span for child nodes: "
                f"{sorted(unaddressed)!r}"
            )
        return examples
