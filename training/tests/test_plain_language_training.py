"""Characterization tests for the plain-language training boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlm_train.feedback import FeedbackBundle, FeedbackVisibility, ScopedAssessment
from rlm_train.generation.generated_text import GeneratedText
from rlm_train.sdpo.score_with_feedback import score_with_feedback
from rlm_train.settings import AssessmentScope, TokenScope
from rlm_train.student import StudentModelInfo, TokenizerInfo, TokenPredictions
from rlm_train.token_selection import (
    SelectedGenerationTokens,
    TokenSelection,
    TokenSelectionResult,
)
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    AnnotationRecord,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    GenerationTokens,
    NodeRole,
    ObjectiveSelection,
    SelectedTokenRange,
    TaskPartition,
)


class RecordingFormatter:
    def __init__(self) -> None:
        self.formatted_messages = None

    def messages(self, prompt):
        return [{"role": "user", "content": str(prompt)}]

    def encode_prompt(self, messages):
        self.formatted_messages = messages
        return (91, 92, 93)


class RecordingStudent:
    def __init__(self, torch) -> None:
        self.model_info = StudentModelInfo(
            component_id="student",
            revision="r1",
            student_id="student:one",
            checkpoint_id="base",
        )
        self.tokenizer_info = TokenizerInfo(component_id="tokenizer", revision="r1")
        self.generator = SimpleNamespace(formatter=RecordingFormatter())
        self.torch = torch
        self.scored = []

    def format_prompt(self, messages):
        return self.generator.formatter.encode_prompt(messages)

    def score_tokens(self, generated_text: GeneratedText, **kwargs):
        self.scored.append((generated_text, kwargs))
        positions = kwargs["positions"]
        return TokenPredictions(
            token_ids=generated_text.token_ids,
            logits=self.torch.zeros((len(positions), 5)),
        )


def attempt_fixture() -> AnnotatedRollout:
    generation = GenerationTokens(
        generation_id="generation-1",
        node_id="root",
        policy_owner="student:one",
        text="answer",
        prompt_token_ids=(1, 2),
        token_ids=(10, 11, 12),
        token_offsets=((0, 2), (2, 4), (4, 6)),
    )
    return AnnotatedRollout(
        rollout_id="attempt-1",
        mode="training",
        task=TaskPartition(task_id="task", public={"question": "q"}),
        policy={"policy_owner": "student:one"},
        execution=ExecutionRecord(
            root_node_id="root",
            nodes=(
                ExecutionNode(
                    node_id="root",
                    role=NodeRole.ROOT,
                    depth=0,
                    policy_owner="student:one",
                    prompt="original prompt",
                ),
                ExecutionNode(
                    node_id="child",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    prompt="helper",
                ),
            ),
            edges=(
                ExecutionEdge(
                    edge_id="edge-1",
                    parent_id="root",
                    child_id="child",
                    kind="plain",
                    question="helper",
                ),
            ),
            events=(
                {
                    "event_type": "student_generation_completed",
                    "event_id": "e1",
                    "sequence_number": 1,
                    "invocation_id": "root",
                    "generation_id": "generation-1",
                },
                {
                    "event_type": "helper_question_generated",
                    "event_id": "e2",
                    "sequence_number": 2,
                    "invocation_id": "root",
                    "subcall_id": "edge-1",
                },
            ),
        ),
        annotations=AnnotationRecord(generations=(generation,)),
    )


def selection_fixture() -> TokenSelectionResult:
    durable = ObjectiveSelection(
        objective="sdpo",
        token_scope=TokenScope.HELPER_QUESTIONS,
        policy_owner="student:one",
        ranges=(
            SelectedTokenRange(
                generation_id="generation-1",
                node_id="root",
                token_start=1,
                token_end=3,
                token_ids=(11, 12),
                reason="test",
            ),
        ),
    )
    return TokenSelectionResult(
        selection=TokenSelection(
            training_method="sdpo",
            attempt_id="attempt-1",
            generations=(
                SelectedGenerationTokens(
                    generation_id="generation-1",
                    positions=(1, 2),
                    text_regions=("helper_question",),
                ),
            ),
        ),
        durable=durable,
        masks={"generation-1": (False, True, True)},
    )


def test_feedback_scoring_reformats_messages_and_preserves_sampled_ids():
    torch = pytest.importorskip("torch")
    student = RecordingStudent(torch)
    assessment = ScopedAssessment(
        assessment_id="assessment-1",
        scope=AssessmentScope.RETROSPECTIVE_LOCAL,
        focal_edge_ids=("edge-1",),
        evidence_node_ids=("root", "child"),
        evidence_event_ids=("e1", "e2"),
        judge_view_fingerprint="a" * 64,
        content={
            "rubric": {
                "improved_question_guidance": "ask for the decisive fact",
                "what_was_missing": "specificity",
            }
        },
        visibility=FeedbackVisibility.RESTRICTED,
        provider="fake",
        model_revision="v1",
        prompt_version="v1",
        cache_key="key",
    )

    result = score_with_feedback(
        student=student,
        attempts=(attempt_fixture(),),
        feedback=FeedbackBundle(local_assessments=(assessment,)),
        selections={"attempt-1": selection_fixture().selection},
        included_text=TokenScope.HELPER_QUESTIONS,
        top_k=2,
    )["attempt-1"]

    generated_text, arguments = student.scored[0]
    assert generated_text.prompt_token_ids == (91, 92, 93)
    assert generated_text.token_ids == (10, 11, 12)
    assert arguments["positions"] == (1, 2)
    assert arguments["with_gradients"] is False
    assert result.selected_token_ids == (11, 12)
    assert "ask for the decisive fact" in str(student.generator.formatter.formatted_messages)
