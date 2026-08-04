"""Protect Colab preflight and concrete structured-judge privacy/provenance."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from rlm.core.trajectory import InvocationKind, InvocationNode, TrajectoryTree

from rlm_train.colab.config import (
    ColabRunConfig,
    DatasetConfig,
    JudgeConfig,
    ModelConfig,
    OptimizationConfig,
    Precision,
)
from rlm_train.colab.runtime import validate_colab_runtime
from rlm_train.experiment import resolve_ablation_preset
from rlm_train.judge import (
    DeterministicFakeStructuredJudgeClient,
    MemoryFeedbackCache,
    OpenAIStructuredJudgeClient,
    PrivilegedJudgeContext,
    StructuredJudgeRequest,
    StructuredOutputTrajectoryJudge,
    TaskContext,
    TeacherFeedbackMode,
)


class FakeCUDA:
    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self.available = available
        self.bf16 = bf16

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def get_device_name(self, device: int = 0) -> str:
        assert device == 0
        return "Fake GPU"


def test_preflight_rejects_cuda_precision_and_missing_api_secret(tmp_path):
    installed = {
        "torch": "test",
        "transformers": "test",
        "peft": "test",
        "accelerate": "test",
        "safetensors": "test",
        "openai": "test",
    }
    config = ColabRunConfig(
        profile="train",
        experiment_preset=None,
        experiment=resolve_ablation_preset("edge_local_sdpo"),
        model=ModelConfig(
            model_id="toy",
            model_revision="v1",
            precision=Precision.BF16,
        ),
        optimization=OptimizationConfig(
            max_optimizer_steps=2,
            sdpo_weight=1.0,
        ),
        dataset=DatasetConfig(rubric="exact_match"),
        judge=JudgeConfig(provider="openai", model="judge-model", model_revision="v1"),
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        validate_colab_runtime(
            config,
            cuda=FakeCUDA(available=False),
            installed_versions=installed,
        )
    with pytest.raises(RuntimeError, match="bf16"):
        validate_colab_runtime(config, cuda=FakeCUDA(bf16=False), installed_versions=installed)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        validate_colab_runtime(
            config,
            cuda=FakeCUDA(),
            environment={},
            base_directory=tmp_path,
            installed_versions=installed,
        )
    result = validate_colab_runtime(
        config,
        cuda=FakeCUDA(),
        environment={"OPENAI_API_KEY": "secret-value"},
        base_directory=tmp_path,
        installed_versions=installed,
    )

    assert result.cuda_device_name == "Fake GPU"
    assert "secret-value" not in repr(result)
    assert result.output_directory.startswith(str(tmp_path))


def make_tree() -> TrajectoryTree:
    return TrajectoryTree(
        trajectory_id="trace",
        nodes=[
            InvocationNode(
                node_id="trace/root/i000",
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="student",
                context="public question",
                response="ask child",
            ),
            InvocationNode(
                node_id="trace/root/i000/c000",
                parent_id="trace/root/i000",
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="student",
                context="public subquestion",
                response="public response",
            ),
        ],
    )


def test_fake_judge_cache_is_projection_equivalent_and_privacy_safe():
    secret = "SENTINEL_PRIVILEGED_ANSWER"
    client = DeterministicFakeStructuredJudgeClient()
    judge = StructuredOutputTrajectoryJudge(
        client,
        judge_version="judge-v1",
        rubric_version="rubric-v1",
        cache=MemoryFeedbackCache(),
    )
    task = TaskContext(
        "task",
        "public question",
        privileged_context=PrivilegedJudgeContext("answers", "v1", {"answer": secret}),
    )

    first = asyncio.run(judge.evaluate(make_tree(), task))
    uncached_projection = first.subcalls[0].to_teacher_view(TeacherFeedbackMode.DIAGNOSTIC)
    second = asyncio.run(judge.evaluate(make_tree(), task))
    cached_projection = second.subcalls[0].to_teacher_view(TeacherFeedbackMode.DIAGNOSTIC)

    assert uncached_projection.model_dump_json() == cached_projection.model_dump_json()
    assert secret not in uncached_projection.model_dump_json()
    assert secret not in json.dumps(client.calls[0].to_dict())
    assert judge.last_cache_key is not None
    assert judge.metrics.cache_hit_count == 1


class FakeResponses:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return SimpleNamespace(
            output_text=self.output,
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


def test_openai_provider_requests_strict_schema_and_records_no_payload():
    output = json.dumps(
        {
            "trajectory_score": 0.5,
            "nodes": [],
            "subcalls": [],
            "judge_version": "judge-v1",
            "rubric_version": "rubric-v1",
            "metadata": {},
        }
    )
    responses = FakeResponses(output)
    client = OpenAIStructuredJudgeClient(
        model="judge-model",
        model_revision="revision-v1",
        prompt_schema_version="schema-v1",
        client=SimpleNamespace(responses=responses),
    )
    request = StructuredJudgeRequest(
        instructions="judge strictly",
        task={"task_id": "task", "prompt": "public"},
        trajectory=make_tree().to_dict(),
        privileged_context={"payload": {"secret": "PRIVATE_SENTINEL"}},
        response_schema={"type": "object"},
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )

    assert asyncio.run(client.complete(request)) == output
    call = client.calls[0].to_dict()

    assert call["input_tokens"] == 11
    assert call["output_tokens"] == 7
    assert "PRIVATE_SENTINEL" not in json.dumps(call)
    assert responses.requests[0]["text"]["format"]["strict"] is True
