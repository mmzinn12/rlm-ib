"""Configurable categorical and full LLM judge contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rlm_train.feedback import FeedbackVisibility
from rlm_train.judge import MemoryJudgeCache, OpenAIJudge, build_judge
from rlm_train.judge.categorical import CategoricalJudgeAssessment
from rlm_train.judge.views import JudgeView
from rlm_train.runtime import ComponentFactory, register_judge_builder
from rlm_train.spec import AssessmentScope, JudgeMode, JudgeSpec, RunSpec, StudentSpec, TokenScope


class QueuedResponses:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0))


class QueuedClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = QueuedResponses(outputs)


def judge_view() -> JudgeView:
    return JudgeView(
        builder_name="test-view-v1",
        scope=AssessmentScope.RETROSPECTIVE_LOCAL,
        focal_node_ids=("root", "child"),
        focal_edge_ids=("edge-1",),
        evidence_node_ids=("root", "child"),
        evidence_event_ids=("event-1", "event-2"),
        upstream_depth=1,
        downstream_depth=1,
        visibility=FeedbackVisibility.RESTRICTED,
        allowed_objectives=frozenset({"sdpo"}),
        allowed_token_scopes=frozenset({TokenScope.HELPER_QUESTIONS}),
        task={"prompt": "Use the supplied context."},
        evidence={
            "question": "Which source resolves the ambiguity?",
            "response": "The second source resolves it.",
        },
    )


def categorical_payload() -> str:
    return json.dumps(
        {
            "significance": "high",
            "novelty": "medium",
            "uncertainty_reduction": "high",
            "evidence_quality": "good",
            "redundant": False,
            "misleading": False,
            "diagnostic": "The answer resolves the focal ambiguity.",
            "information_revealed": ["The second source contains the distinguishing fact."],
            "rubric": {
                "information_revealed": ["The second source contains the distinguishing fact."],
                "what_was_missing": "The specific publication year.",
                "redundant_with_context": False,
                "misleading_or_invalid": False,
                "why_it_mattered": "It disambiguated the two candidate sources.",
                "improved_question_guidance": "Name the source and ask for its publication year.",
                "rationale": "The question narrowed the search but omitted the decisive detail.",
            },
        }
    )


def full_payload(*, significance: float = 0.75) -> str:
    return json.dumps(
        {
            "information_significance": significance,
            "novelty": 0.5,
            "uncertainty_reduction": 0.75,
            "evidence_quality": 1.0,
            "redundant": False,
            "misleading": False,
            "diagnostic": "The question retrieved relevant evidence.",
            "information_revealed": ["A relevant fact."],
            "rationale": "The response directly reduced uncertainty.",
            "rubric": {
                "information_revealed": ["A relevant fact."],
                "what_was_missing": "A corroborating second source.",
                "redundant_with_context": False,
                "misleading_or_invalid": False,
                "why_it_mattered": "It reduced uncertainty about the focal claim.",
                "improved_question_guidance": "Ask for a second independent source.",
                "rationale": "The response directly reduced uncertainty.",
            },
        }
    )


def test_categorical_assessment_maps_enums_to_bounded_scores():
    assessment = CategoricalJudgeAssessment.model_validate_json(categorical_payload())

    assert assessment.normalized_content()["information_significance"] == 1.0
    assert assessment.normalized_content()["novelty"] == 0.6
    assert assessment.normalized_content()["uncertainty_reduction"] == 1.0
    with pytest.raises(ValidationError):
        CategoricalJudgeAssessment.model_validate(
            {
                **json.loads(categorical_payload()),
                "uncertainty_reduction": 4,
            }
        )


def test_categorical_openai_judge_returns_cached_scoped_assessment():
    client = QueuedClient([categorical_payload()])
    cache = MemoryJudgeCache()
    spec = JudgeSpec(
        provider="openai",
        model="Qwen/Qwen2.5-7B-Instruct:together",
        model_revision="revision",
        mode=JudgeMode.CATEGORICAL,
    )
    judge = OpenAIJudge(spec, client=client, cache=cache)

    assert "original-task relevance gate" in judge.instructions
    assert "task-relevance-v2" in judge.cache_prompt_version

    first = judge.assess(judge_view())
    second = judge.assess(judge_view())

    assert first == second
    assert first.provider == "openai:categorical"
    assert first.content["information_significance"] == 1.0
    assert len(client.responses.calls) == 1
    schema = client.responses.calls[0]["text"]["format"]["schema"]
    assert schema["properties"]["significance"]["$ref"].endswith(
        "/InformationSignificance"
    )


def test_full_openai_judge_retries_out_of_range_numeric_output():
    client = QueuedClient([full_payload(significance=3.0), full_payload()])
    spec = JudgeSpec(
        provider="openai",
        model="Qwen/Qwen2.5-7B-Instruct:together",
        model_revision="revision",
        mode=JudgeMode.FULL,
        max_attempts=2,
    )
    judge = OpenAIJudge(spec, client=client)

    assessment = judge.assess(judge_view())

    assert assessment.provider == "openai:full"
    assert assessment.content["information_significance"] == 0.75
    assert len(client.responses.calls) == 2
    retry_payload = json.loads(client.responses.calls[1]["input"])
    assert "less than or equal to 1" in retry_payload["previous_validation_error"]


def test_run_spec_selects_mode_and_runtime_factory_builds_configured_judge():
    client = QueuedClient([categorical_payload()])
    spec = RunSpec(
        student=StudentSpec(model_id="student"),
        judge=JudgeSpec(
            provider="openai",
            model="Qwen/Qwen2.5-7B-Instruct:together",
            model_revision="revision",
            mode="categorical",
            base_url="https://router.huggingface.co/v1",
        ),
    )
    factory = ComponentFactory()
    register_judge_builder(factory, client=client)

    components = factory.resolve(spec)

    assert spec.judge.mode is JudgeMode.CATEGORICAL
    assert isinstance(components.judge, OpenAIJudge)
    assert str(components.judge.spec.base_url) == "https://router.huggingface.co/v1"
    assert build_judge(JudgeSpec()).assess(judge_view()).provider == "fake"


def test_judge_spec_rejects_markdown_instead_of_accepting_it_as_a_base_url():
    with pytest.raises(ValidationError):
        JudgeSpec(
            provider="openai",
            model="judge",
            base_url="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)",
        )
