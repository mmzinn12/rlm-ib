"""Validate the node-addressable feedback contract used by tree-aware SDPO.

Purpose:
    Turn judge output into strict, typed feedback for routing, calls, reasoning,
    aggregation, final answers, and subcall information significance.
Implementation:
    Frozen-shape Pydantic models reject unknown fields, enforce score bounds, and check
    uniqueness and trajectory-node references before feedback reaches the trainer.
Inputs:
    Parsed JSON or Python values returned by a trajectory judge.
Outputs:
    Validated ``TrajectoryFeedback`` and component feedback models.
Example:
    ``feedback = TrajectoryFeedback(trajectory_score=1.0, judge_version="j1", rubric_version="r1")``
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base feedback model that rejects fields outside the declared schema."""

    model_config = ConfigDict(extra="forbid")


class TeacherFeedbackMode(StrEnum):
    """Select the single judge-to-teacher projection used for one run."""

    SCALAR = "scalar"
    DIAGNOSTIC = "diagnostic"
    FACTUAL = "factual"


class RoutingFeedback(StrictModel):
    """Assess whether recursion was selected, ordered, and stopped appropriately.

    Attributes:
        quality: Coarse routing assessment.
        missing_calls: Useful questions or calls that were not made.
        redundant_calls: Calls whose expected information was already available.
        repair_direction: Concise guidance for a better routing decision.
    """

    quality: Literal["poor", "mixed", "good"]
    missing_calls: list[str] = Field(default_factory=list)
    redundant_calls: list[str] = Field(default_factory=list)
    repair_direction: str = ""


class CallFeedback(StrictModel):
    """Assess construction of a generated recursive call.

    Attributes:
        invalid_slice: Whether the selected range or object was invalid.
        wrong_variable: Whether the action referenced the wrong context variable.
        repair_direction: Concise guidance for constructing a better call.
    """

    invalid_slice: bool = False
    wrong_variable: bool = False
    repair_direction: str = ""


class ReasoningFeedback(StrictModel):
    """Describe localized reasoning errors and supported constraints for one node.

    Attributes:
        first_error_step: Optional first erroneous reasoning-step index.
        error_spans: Human-readable descriptions or excerpts locating errors.
        constraint_results: Named scientific or task constraints and whether they pass.
        unsupported_claims: Claims not justified by evidence visible to the node.
        repair_direction: Guidance for revising the reasoning.
        do_not_claim: Conclusions the available evidence cannot support.
    """

    first_error_step: int | None = None
    error_spans: list[str] = Field(default_factory=list)
    constraint_results: dict[str, bool] = Field(default_factory=dict)
    unsupported_claims: list[str] = Field(default_factory=list)
    repair_direction: str = ""
    do_not_claim: list[str] = Field(default_factory=list)


class AggregationFeedback(StrictModel):
    """Assess how a parent interpreted and combined child results.

    Attributes:
        ignored_children: Relevant child node IDs or results that were omitted.
        misinterpreted_children: Child results represented incorrectly.
        conflicts_unresolved: Evidence conflicts left unresolved or unacknowledged.
        repair_direction: Guidance for a better synthesis.
    """

    ignored_children: list[str] = Field(default_factory=list)
    misinterpreted_children: list[str] = Field(default_factory=list)
    conflicts_unresolved: list[str] = Field(default_factory=list)
    repair_direction: str = ""


class FinalAnswerFeedback(StrictModel):
    """Assess the final answer independently of subcall information value.

    Attributes:
        outcome: Overall correctness category.
        errors: Specific correctness, support, or calibration failures.
        repair_direction: Guidance for producing a better final answer.
    """

    outcome: Literal["incorrect", "partial", "correct"]
    errors: list[str] = Field(default_factory=list)
    repair_direction: str = ""


class QuestionTeacherFeedback(StrictModel):
    """Expose only bounded scalar edge feedback to a question scorer.

    Attributes:
        parent_node_id: Node that generated the scored question.
        child_node_id: Child invocation whose result established information value.
        information_significance: Signed information-value signal in ``[-1, 1]``.
        uncertainty_reduction: Fraction of live uncertainty resolved by the result.
        novelty: Degree to which the result added information absent from the parent.
        evidence_quality: Reliability and relevance of the returned evidence.
        redundant: Whether the result merely repeated the parent context.
        misleading: Whether the result was invalid or actively misleading.
        mode: Scalar projection selected once for the enclosing training run.

    Example:
        ``view = information_feedback.to_teacher_view()``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal[TeacherFeedbackMode.SCALAR] = TeacherFeedbackMode.SCALAR
    projector_name: Literal["edge_local_question_feedback"] = "edge_local_question_feedback"
    projector_version: str = Field(min_length=1)
    parent_node_id: str = Field(min_length=1)
    child_node_id: str = Field(min_length=1)
    information_significance: float = Field(ge=-1.0, le=1.0)
    uncertainty_reduction: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    redundant: bool = False
    misleading: bool = False


class DiagnosticQuestionTeacherFeedback(QuestionTeacherFeedback):
    """Add one bounded edge-local defect description without concrete solution facts."""

    mode: Literal[TeacherFeedbackMode.DIAGNOSTIC] = TeacherFeedbackMode.DIAGNOSTIC
    diagnostic: str = Field(min_length=1, max_length=500)


class FactualQuestionTeacherFeedback(QuestionTeacherFeedback):
    """Expose concrete edge facts as an intentionally information-rich control arm."""

    mode: Literal[TeacherFeedbackMode.FACTUAL] = TeacherFeedbackMode.FACTUAL
    diagnostic: str | None = Field(default=None, max_length=500)
    information_revealed: tuple[str, ...] = ()
    reference_aligned_correction: str | None = None

    @model_validator(mode="after")
    def validate_factual_payload(self) -> FactualQuestionTeacherFeedback:
        """Reject blank factual values while allowing an empty factual assessment."""
        if self.diagnostic is not None and not self.diagnostic.strip():
            raise ValueError("diagnostic must not be blank when supplied")
        if any(not fact.strip() for fact in self.information_revealed):
            raise ValueError("information_revealed cannot contain blank values")
        if (
            self.reference_aligned_correction is not None
            and not self.reference_aligned_correction.strip()
        ):
            raise ValueError("reference_aligned_correction must not be blank")
        return self


class InformationValueFeedback(StrictModel):
    """Score a subcall by information revealed, never by outcome contribution.

    Attributes:
        parent_node_id: Node that generated the call or question.
        child_node_id: Child invocation that returned the information.
        information_significance: Signed reward/penalty placeholder in ``[-1, 1]``.
        novelty: Degree to which the result was new relative to the parent context.
        uncertainty_reduction: Degree to which the result resolved live uncertainty.
        evidence_quality: Reliability and relevance of the returned evidence.
        information_revealed: Concrete facts or distinctions learned from the call.
        redundant_with_parent_context: Whether the result repeated known information.
        misleading_or_invalid: Whether the result was unusable or actively misleading.
        rationale: Explanation of the information-value assessment.

    Example:
        ``InformationValueFeedback(parent_node_id="root", child_node_id="child", information_significance=0.8, novelty=0.7, uncertainty_reduction=0.6, evidence_quality=0.9)``
    """

    parent_node_id: str
    child_node_id: str
    information_significance: float = Field(ge=-1.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    uncertainty_reduction: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    information_revealed: list[str] = Field(default_factory=list)
    redundant_with_parent_context: bool = False
    misleading_or_invalid: bool = False
    edge_local_diagnostic: str = "No edge-local reasoning defect was identified."
    reference_aligned_correction: str = ""
    rationale: str = ""

    def to_teacher_view(
        self,
        mode: TeacherFeedbackMode = TeacherFeedbackMode.DIAGNOSTIC,
        *,
        projector_version: str = "v1",
    ) -> QuestionTeacherFeedback:
        """Create the restricted feedback boundary used for question-token scoring.

        The conversion intentionally drops judge rationale, global reward/correctness,
        reference questions, and any outcome-contribution metadata.

        Returns:
            A frozen ``QuestionTeacherFeedback`` containing only fields attributable
            to this parent-child information edge.

        Example:
            ``teacher_feedback = feedback.to_teacher_view()``
        """
        mode = TeacherFeedbackMode(mode)
        if not projector_version.strip():
            raise ValueError("projector_version must not be blank")
        diagnostic = None
        information_revealed: tuple[str, ...] = ()
        correction = None
        if mode in (TeacherFeedbackMode.DIAGNOSTIC, TeacherFeedbackMode.FACTUAL):
            diagnostic = self.edge_local_diagnostic
        if mode is TeacherFeedbackMode.FACTUAL:
            information_revealed = tuple(self.information_revealed)
            correction = self.reference_aligned_correction or None
        common = {
            "projector_version": projector_version,
            "parent_node_id": self.parent_node_id,
            "child_node_id": self.child_node_id,
            "information_significance": self.information_significance,
            "uncertainty_reduction": self.uncertainty_reduction,
            "novelty": self.novelty,
            "evidence_quality": self.evidence_quality,
            "redundant": self.redundant_with_parent_context,
            "misleading": self.misleading_or_invalid,
        }
        if mode is TeacherFeedbackMode.SCALAR:
            return QuestionTeacherFeedback(**common)
        if mode is TeacherFeedbackMode.DIAGNOSTIC:
            return DiagnosticQuestionTeacherFeedback(diagnostic=diagnostic, **common)
        return FactualQuestionTeacherFeedback(
            diagnostic=diagnostic,
            information_revealed=information_revealed,
            reference_aligned_correction=correction,
            **common,
        )


class NodeFeedback(StrictModel):
    """Group all applicable component feedback for one trajectory node.

    Attributes:
        node_id: Stable node ID from the traced trajectory.
        routing_feedback: Optional recursion-selection feedback.
        call_feedback: Optional recursive-call construction feedback.
        reasoning_feedback: Optional node-local reasoning feedback.
        aggregation_feedback: Optional parent synthesis feedback.
    """

    node_id: str
    routing_feedback: RoutingFeedback | None = None
    call_feedback: CallFeedback | None = None
    reasoning_feedback: ReasoningFeedback | None = None
    aggregation_feedback: AggregationFeedback | None = None


class TrajectoryFeedback(StrictModel):
    """Represent the complete structured output expected from a trajectory judge.

    Attributes:
        trajectory_score: Optional global evaluation score retained for reporting.
        final_answer_feedback: Feedback applied to final-answer spans.
        nodes: Node-addressable component feedback.
        subcalls: Edge-addressable information-significance feedback.
        judge_version: Evaluator implementation or prompt version.
        rubric_version: Feedback schema/rubric version.
        metadata: Additional evaluation metadata not consumed by the core compiler.

    Example:
        ``TrajectoryFeedback(trajectory_score=0.0, judge_version="j1", rubric_version="r1")``
    """

    trajectory_score: float
    final_answer_feedback: FinalAnswerFeedback | None = None
    nodes: list[NodeFeedback] = Field(default_factory=list)
    subcalls: list[InformationValueFeedback] = Field(default_factory=list)
    judge_version: str
    rubric_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_references(self) -> TrajectoryFeedback:
        """Reject duplicate node and child-edge assessments.

        Returns:
            This validated model instance.

        Raises:
            ValueError: If a node has multiple node-feedback entries or a child has
                multiple information-value assessments.
        """
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node feedback IDs must be unique")
        child_ids = [subcall.child_node_id for subcall in self.subcalls]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("each child node may have only one information-value assessment")
        return self

    def validate_node_ids(self, known_node_ids: set[str]) -> None:
        """Ensure all feedback references nodes present in a trajectory.

        Args:
            known_node_ids: Complete set of node IDs from the associated trajectory.

        Returns:
            ``None`` after validation succeeds.

        Raises:
            ValueError: If node feedback or subcall feedback references an unknown ID.
        """
        referenced = {node.node_id for node in self.nodes}
        for subcall in self.subcalls:
            referenced.add(subcall.parent_node_id)
            referenced.add(subcall.child_node_id)
        unknown = referenced - known_node_ids
        if unknown:
            raise ValueError(f"feedback references unknown node IDs: {sorted(unknown)!r}")
