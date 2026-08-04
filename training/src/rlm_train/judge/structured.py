"""Run a bounded-retry structured trajectory judge behind a provider adapter.

Purpose:
    Supply one concrete judge implementation without coupling the training package to
    a specific model vendor.
Implementation:
    A small client protocol receives a strict request, raw responses are validated as
    ``TrajectoryFeedback``, invalid responses are retried within a fixed bound, and
    successful feedback is stored in a content-addressed cache.
Inputs:
    A completed trajectory, public task context, optional privileged judge context,
    judge/rubric versions, and a structured-output client.
Outputs:
    Strict node-addressable feedback plus observable execution metrics.
Example:
    ``judge = StructuredOutputTrajectoryJudge(client, judge_version="j1", rubric_version="r1")``
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from pydantic import ValidationError
from rlm.core.trajectory import TrajectoryTree

from rlm_train.judge.base import TaskContext
from rlm_train.judge.cache import FeedbackCache, make_trajectory_feedback_cache_key
from rlm_train.judge.prompts import build_judge_instructions
from rlm_train.judge.schema import TrajectoryFeedback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuredJudgeRequest:
    """Carry one provider-neutral structured-output request.

    The privileged context is excluded from ``repr`` and materialized only by
    :meth:`to_payload`, keeping routine logs and diagnostics payload-free.
    """

    instructions: str
    task: dict[str, Any]
    trajectory: dict[str, Any]
    response_schema: dict[str, Any]
    judge_version: str
    rubric_version: str
    attempt: int = 1
    previous_error: str | None = None
    privileged_context: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_payload(self) -> dict[str, Any]:
        """Return the complete request payload for the configured judge client."""
        return {
            "instructions": self.instructions,
            "task": self.task,
            "trajectory": self.trajectory,
            "privileged_context": self.privileged_context,
            "response_schema": self.response_schema,
            "judge_version": self.judge_version,
            "rubric_version": self.rubric_version,
            "attempt": self.attempt,
            "previous_error": self.previous_error,
        }


class StructuredJudgeClient(Protocol):
    """Adapt a provider SDK to the structured trajectory-judge request."""

    async def complete(self, request: StructuredJudgeRequest) -> Any:
        """Return ``TrajectoryFeedback``, a compatible mapping, or a JSON string."""
        ...


@dataclass
class JudgeExecutionMetrics:
    """Count cache, validation, retry, and success outcomes."""

    request_count: int = 0
    cache_hit_count: int = 0
    invalid_response_count: int = 0
    retry_count: int = 0
    repaired_response_count: int = 0
    exhausted_response_count: int = 0
    success_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return tracker-ready metric names and counts."""
        return {
            "judge/request_count": self.request_count,
            "judge/cache_hit_count": self.cache_hit_count,
            "judge/invalid_response_count": self.invalid_response_count,
            "judge/retry_count": self.retry_count,
            "judge/repaired_response_count": self.repaired_response_count,
            "judge/exhausted_response_count": self.exhausted_response_count,
            "judge/success_count": self.success_count,
        }


class JudgeResponseError(RuntimeError):
    """Report exhaustion of the structured-response validation budget."""


class StructuredOutputTrajectoryJudge:
    """Evaluate trajectories with strict schema validation and bounded retries."""

    def __init__(
        self,
        client: StructuredJudgeClient,
        *,
        judge_version: str,
        rubric_version: str,
        task_instructions: str = "",
        max_attempts: int = 2,
        cache: FeedbackCache | None = None,
    ) -> None:
        """Configure one versioned judge and optional persistent cache."""
        if not judge_version.strip() or not rubric_version.strip():
            raise ValueError("judge_version and rubric_version must not be blank")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.client = client
        self.judge_version = judge_version
        self.rubric_version = rubric_version
        self.instructions = build_judge_instructions(task_instructions)
        self.max_attempts = max_attempts
        self.cache = cache
        self.metrics = JudgeExecutionMetrics()
        self.last_cache_key: str | None = None

    async def evaluate(
        self,
        trajectory: TrajectoryTree,
        task: TaskContext,
    ) -> TrajectoryFeedback:
        """Evaluate one completed rollout without exposing privileged data elsewhere."""
        trajectory.validate()
        known_node_ids = {node.node_id for node in trajectory.nodes}
        descriptor = task.privileged_descriptor()
        cache_key = make_trajectory_feedback_cache_key(
            task=task.public_payload(),
            trajectory=trajectory.to_dict(),
            privileged_context_fingerprint=(descriptor.fingerprint if descriptor else None),
            judge_version=self.judge_version,
            rubric_version=self.rubric_version,
        )
        self.last_cache_key = cache_key
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached.validate_node_ids(known_node_ids)
                self.metrics.cache_hit_count += 1
                return cached

        task_payload = task.judge_payload()
        privileged_context = task_payload.pop("privileged_context")
        base_request = StructuredJudgeRequest(
            instructions=self.instructions,
            task=task_payload,
            trajectory=trajectory.to_dict(),
            privileged_context=privileged_context,
            response_schema=TrajectoryFeedback.model_json_schema(),
            judge_version=self.judge_version,
            rubric_version=self.rubric_version,
        )
        previous_error: str | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = replace(
                base_request,
                attempt=attempt,
                previous_error=previous_error,
            )
            self.metrics.request_count += 1
            raw = await self.client.complete(request)
            try:
                feedback = parse_feedback(raw)
                if feedback.judge_version != self.judge_version:
                    raise ValueError("judge response version does not match configured judge")
                if feedback.rubric_version != self.rubric_version:
                    raise ValueError("judge response rubric does not match configured rubric")
                feedback.validate_node_ids(known_node_ids)
            except (TypeError, ValueError, ValidationError) as exc:
                self.metrics.invalid_response_count += 1
                last_error = exc
                previous_error = str(exc)
                if attempt < self.max_attempts:
                    self.metrics.retry_count += 1
                    logger.warning(
                        "rejected invalid structured judge response; retrying (attempt %s/%s)",
                        attempt,
                        self.max_attempts,
                    )
                    continue
                self.metrics.exhausted_response_count += 1
                logger.error(
                    "rejected invalid structured judge response; attempts exhausted (%s/%s)",
                    attempt,
                    self.max_attempts,
                )
                break
            if self.cache is not None:
                self.cache.put(cache_key, feedback)
            if attempt > 1:
                self.metrics.repaired_response_count += 1
            self.metrics.success_count += 1
            return feedback
        raise JudgeResponseError(
            f"structured judge failed validation after {self.max_attempts} attempts"
        ) from last_error


def parse_feedback(raw: Any) -> TrajectoryFeedback:
    """Normalize supported provider response shapes into the strict schema."""
    if isinstance(raw, TrajectoryFeedback):
        return raw
    if isinstance(raw, str):
        return TrajectoryFeedback.model_validate_json(raw)
    if isinstance(raw, Mapping):
        return TrajectoryFeedback.model_validate(dict(raw))
    raise TypeError("structured judge client returned an unsupported response type")


__all__ = [
    "JudgeExecutionMetrics",
    "JudgeResponseError",
    "StructuredJudgeClient",
    "StructuredJudgeRequest",
    "StructuredOutputTrajectoryJudge",
]
