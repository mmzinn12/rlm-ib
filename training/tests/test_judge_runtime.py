"""Verify privileged judge isolation, structured retries, and persistent caching.

Purpose:
    Protect the code-level boundary that makes privileged evidence visible only to the
    judge request while keeping artifacts, cache keys, logs, and teacher views clean.
Implementation:
    A fake structured client returns one invalid response followed by valid feedback;
    SQLite persistence and retry metrics are then inspected.
Inputs:
    A minimal trajectory, a sealed privileged payload, and deterministic fake outputs.
Outputs:
    Assertions over isolation, retries, cache hits, and persistent cache contents.
Example:
    Run ``pytest training/tests/test_judge_runtime.py`` from the repository root.
"""

import json
from typing import Any

import pytest
from rlm.core.trajectory import InvocationKind, InvocationNode, TrajectoryTree

from rlm_train.judge.base import TaskContext
from rlm_train.judge.cache import SQLiteFeedbackCache
from rlm_train.judge.context import PrivilegedJudgeContext
from rlm_train.judge.privileged import PrivilegedContextTrajectoryJudge
from rlm_train.judge.schema import TrajectoryFeedback
from rlm_train.judge.structured import (
    JudgeResponseError,
    StructuredJudgeRequest,
    StructuredOutputTrajectoryJudge,
)


class FakeStructuredClient:
    """Return queued provider responses and retain payloads for boundary assertions."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredJudgeRequest] = []

    async def complete(self, request: StructuredJudgeRequest) -> Any:
        """Return the next queued response."""
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("structured client should not have been called")
        return self.responses.pop(0)


class FixedPrivilegedContextProvider:
    """Return one preconfigured privileged context and retain lookup identity."""

    def __init__(self, context: PrivilegedJudgeContext | None) -> None:
        self.context = context
        self.task_ids: list[str] = []

    async def get_context(
        self,
        *,
        task_id: str,
        trajectory: TrajectoryTree,
    ) -> PrivilegedJudgeContext | None:
        """Return context after validating the completed trajectory."""
        trajectory.validate()
        self.task_ids.append(task_id)
        return self.context


def make_tree() -> TrajectoryTree:
    """Build one valid root-only rollout."""
    return TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id="run/root/i000",
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="student",
                context="public student context",
                response="final response",
            )
        ],
    )


def make_feedback() -> TrajectoryFeedback:
    """Build one version-matched structured response."""
    return TrajectoryFeedback(
        trajectory_score=0.5,
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )


def test_privileged_context_is_immutable_and_absent_from_public_payloads():
    secret = "reference answer that the student must never see"
    privileged = PrivilegedJudgeContext(
        "answer-key",
        "v1",
        {"reference": secret},
        metadata={"split": "validation"},
    )
    task = TaskContext("task-1", "public prompt", privileged_context=privileged)

    assert secret not in repr(privileged)
    assert secret not in json.dumps(task.public_payload())
    assert secret not in json.dumps(privileged.descriptor().to_dict())
    assert task.judge_payload()["privileged_context"]["payload"]["reference"] == secret
    with pytest.raises(AttributeError, match="immutable"):
        privileged.version = "v2"


@pytest.mark.asyncio
async def test_privileged_provider_attaches_context_only_for_downstream_judge():
    privileged = PrivilegedJudgeContext("answer-key", "v1", {"reference": "secret"})
    provider = FixedPrivilegedContextProvider(privileged)
    client = FakeStructuredClient([make_feedback().model_dump()])
    downstream = StructuredOutputTrajectoryJudge(
        client,
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )
    judge = PrivilegedContextTrajectoryJudge(downstream, provider)
    task = TaskContext("task-1", "public prompt")

    assert await judge.evaluate(make_tree(), task) == make_feedback()
    assert provider.task_ids == ["task-1"]
    assert task.privileged_context is None
    assert client.requests[0].privileged_context == privileged.to_judge_payload()


@pytest.mark.asyncio
async def test_structured_judge_retries_and_reuses_persistent_cache(tmp_path):
    secret = "privileged-control-result"
    privileged = PrivilegedJudgeContext("controls", "v3", {"expected": secret})
    task = TaskContext("task-1", "public prompt", privileged_context=privileged)
    cache_path = tmp_path / "feedback.sqlite3"
    client = FakeStructuredClient(
        [
            {"trajectory_score": 0.0},
            make_feedback().model_dump(),
        ]
    )
    judge = StructuredOutputTrajectoryJudge(
        client,
        judge_version="judge-v1",
        rubric_version="rubric-v1",
        max_attempts=2,
        cache=SQLiteFeedbackCache(cache_path),
    )

    feedback = await judge.evaluate(make_tree(), task)
    cached = await judge.evaluate(make_tree(), task)

    assert feedback == cached == make_feedback()
    assert len(client.requests) == 2
    assert client.requests[1].attempt == 2
    assert client.requests[1].previous_error
    assert client.requests[1].to_payload()["privileged_context"]["payload"]["expected"] == secret
    assert judge.metrics.to_dict() == {
        "judge/request_count": 2,
        "judge/cache_hit_count": 1,
        "judge/invalid_response_count": 1,
        "judge/retry_count": 1,
        "judge/repaired_response_count": 1,
        "judge/exhausted_response_count": 0,
        "judge/success_count": 1,
    }
    assert secret.encode() not in cache_path.read_bytes()

    restarted_client = FakeStructuredClient([])
    restarted = StructuredOutputTrajectoryJudge(
        restarted_client,
        judge_version="judge-v1",
        rubric_version="rubric-v1",
        cache=SQLiteFeedbackCache(cache_path),
    )

    assert await restarted.evaluate(make_tree(), task) == make_feedback()
    assert restarted.metrics.cache_hit_count == 1
    assert restarted_client.requests == []


@pytest.mark.asyncio
async def test_structured_judge_reports_exhausted_invalid_responses():
    client = FakeStructuredClient([{}, {}])
    judge = StructuredOutputTrajectoryJudge(
        client,
        judge_version="judge-v1",
        rubric_version="rubric-v1",
        max_attempts=2,
    )

    with pytest.raises(JudgeResponseError, match="after 2 attempts"):
        await judge.evaluate(make_tree(), TaskContext("task-1", "public prompt"))

    assert judge.metrics.invalid_response_count == 2
    assert judge.metrics.retry_count == 1
    assert judge.metrics.repaired_response_count == 0
    assert judge.metrics.exhausted_response_count == 1
