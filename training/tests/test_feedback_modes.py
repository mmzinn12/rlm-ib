"""Verify deterministic feedback projections and privileged-data boundaries."""

import json

import pytest
from pydantic import ValidationError

from rlm_train.judge import (
    DiagnosticQuestionTeacherFeedback,
    InformationValueFeedback,
    QuestionTeacherFeedback,
    TeacherFeedbackMode,
)


def make_private_assessment(secret: str) -> InformationValueFeedback:
    """Build one rich judge-private assessment containing sentinel payloads."""
    return InformationValueFeedback(
        parent_node_id="root",
        child_node_id="child",
        information_significance=0.7,
        novelty=0.8,
        uncertainty_reduction=0.6,
        evidence_quality=0.9,
        redundant_with_parent_context=False,
        misleading_or_invalid=True,
        edge_local_diagnostic="The parity question used the converse incorrectly.",
        information_revealed=[secret],
        reference_aligned_correction=f"reference correction: {secret}",
        rationale=f"private rationale: {secret}",
    )


def test_scalar_and_diagnostic_projections_structurally_exclude_factual_payloads():
    secret = "SENTINEL_REFERENCE_ANSWER_731"
    assessment = make_private_assessment(secret)

    scalar = assessment.to_teacher_view(TeacherFeedbackMode.SCALAR)
    diagnostic = assessment.to_teacher_view(TeacherFeedbackMode.DIAGNOSTIC)

    assert scalar == assessment.to_teacher_view(TeacherFeedbackMode.SCALAR)
    assert diagnostic == assessment.to_teacher_view(TeacherFeedbackMode.DIAGNOSTIC)
    for projected in (scalar, diagnostic):
        payload = projected.model_dump(mode="json")
        encoded = json.dumps(payload)
        assert "information_revealed" not in payload
        assert "reference_aligned_correction" not in payload
        assert "rationale" not in payload
        assert "trajectory_score" not in payload
        assert secret not in encoded
        assert payload["projector_name"] == "edge_local_question_feedback"
        assert payload["projector_version"] == "v1"
    assert "diagnostic" not in scalar.model_dump()
    assert diagnostic.diagnostic == "The parity question used the converse incorrectly."


def test_factual_projection_is_an_explicit_information_rich_control():
    secret = "SENTINEL_FACT"
    factual = make_private_assessment(secret).to_teacher_view(TeacherFeedbackMode.FACTUAL)

    assert factual.mode is TeacherFeedbackMode.FACTUAL
    assert factual.information_revealed == (secret,)
    assert secret in factual.model_dump_json()


def test_restricted_models_reject_fields_from_richer_modes():
    common = {
        "projector_version": "v1",
        "parent_node_id": "root",
        "child_node_id": "child",
        "information_significance": 0.0,
        "uncertainty_reduction": 0.0,
        "novelty": 0.0,
        "evidence_quality": 0.0,
    }
    with pytest.raises(ValidationError, match="information_revealed"):
        QuestionTeacherFeedback(**common, information_revealed=["secret"])
    with pytest.raises(ValidationError, match="reference_aligned_correction"):
        DiagnosticQuestionTeacherFeedback(
            **common,
            diagnostic="local defect only",
            reference_aligned_correction="answer",
        )
