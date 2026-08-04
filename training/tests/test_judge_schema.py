"""Verify strict judge schemas and information-significance semantics.

Purpose:
    Ensure subcalls are evaluated by revealed information rather than answer outcome,
    and that invalid scores or node references fail validation.
Implementation:
    Pydantic models are constructed from valid and invalid payloads and their serialized
    field names are compared with the documented feedback contract.
Inputs:
    Synthetic feedback values and known trajectory node IDs.
Outputs:
    Pytest assertions and expected ``ValidationError``/``ValueError`` exceptions.
Example:
    Run ``pytest training/tests/test_judge_schema.py`` from the repository root.
"""

import pytest
from pydantic import ValidationError

from rlm_train.judge.prompts import SUBCALL_INFORMATION_VALUE_INSTRUCTIONS
from rlm_train.judge.schema import (
    InformationValueFeedback,
    NodeFeedback,
    RoutingFeedback,
    TrajectoryFeedback,
)


def test_subcall_feedback_scores_information_not_answer_contribution():
    feedback = InformationValueFeedback(
        parent_node_id="root",
        child_node_id="child",
        information_significance=0.8,
        novelty=0.9,
        uncertainty_reduction=0.7,
        evidence_quality=0.6,
        information_revealed=["The assay effect disappears under the control condition."],
    )

    assert feedback.information_significance == 0.8
    assert "final answer" in SUBCALL_INFORMATION_VALUE_INSTRUCTIONS
    assert "outcome_contribution" not in InformationValueFeedback.model_fields


def test_information_value_bounds_are_validated():
    with pytest.raises(ValidationError):
        InformationValueFeedback(
            parent_node_id="root",
            child_node_id="child",
            information_significance=2.0,
            novelty=0.5,
            uncertainty_reduction=0.5,
            evidence_quality=0.5,
        )


def test_feedback_rejects_unknown_node_references():
    feedback = TrajectoryFeedback(
        trajectory_score=0.0,
        subcalls=[
            InformationValueFeedback(
                parent_node_id="root",
                child_node_id="missing",
                information_significance=0.0,
                novelty=0.0,
                uncertainty_reduction=0.0,
                evidence_quality=0.0,
            )
        ],
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    with pytest.raises(ValueError, match="unknown node"):
        feedback.validate_node_ids({"root"})


def test_feedback_dump_matches_the_node_addressable_readme_contract():
    feedback = TrajectoryFeedback(
        trajectory_score=0.0,
        nodes=[
            NodeFeedback(
                node_id="root",
                routing_feedback=RoutingFeedback(quality="mixed"),
            )
        ],
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    dumped = feedback.model_dump()
    assert "final_answer_feedback" in dumped
    assert "routing_feedback" in dumped["nodes"][0]


def test_information_value_teacher_view_excludes_rationale_and_outcome_fields():
    feedback = InformationValueFeedback(
        parent_node_id="root",
        child_node_id="child",
        information_significance=0.7,
        novelty=0.8,
        uncertainty_reduction=0.6,
        evidence_quality=0.9,
        redundant_with_parent_context=True,
        information_revealed=["fact"],
        rationale="privileged judge reasoning",
    )

    teacher_view = feedback.to_teacher_view().model_dump()

    assert teacher_view["redundant"] is True
    assert "information_revealed" not in teacher_view
    assert "reference_aligned_correction" not in teacher_view
    assert "rationale" not in teacher_view
    assert "trajectory_score" not in teacher_view
