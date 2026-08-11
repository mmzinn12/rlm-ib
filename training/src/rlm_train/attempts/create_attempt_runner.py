"""Construct the full-RLM attempt runner from run settings and a student client."""

from __future__ import annotations

from typing import Any

from rlm.clients.base_lm import BaseLM

from rlm_train.attempts.attempt_runner import RLMAttemptRunner
from rlm_train.settings import RunSettings


def create_attempt_runner(
    settings: RunSettings,
    *,
    student_client: BaseLM,
    backend: str = "openai",
    environment_kwargs: dict[str, Any] | None = None,
) -> RLMAttemptRunner:
    return RLMAttemptRunner(
        student_client=student_client,
        student_id=settings.student.resolved_policy_owner,
        spec=settings.rollout,
        backend=backend,
        environment_kwargs=environment_kwargs,
    )


__all__ = ["create_attempt_runner"]
