"""Concrete API and deterministic fake clients for the structured judge protocol."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from rlm_train.judge.schema import (
    FinalAnswerFeedback,
    InformationValueFeedback,
    TrajectoryFeedback,
)
from rlm_train.judge.structured import StructuredJudgeRequest


@dataclass(frozen=True)
class JudgeProviderCall:
    """Record provider/model/latency/usage without prompts, secrets, or private context."""

    provider: str
    model: str
    model_revision: str
    prompt_schema_version: str
    request_fingerprint: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return safe public call provenance."""
        return {
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "prompt_schema_version": self.prompt_schema_version,
            "request_fingerprint": self.request_fingerprint,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class OpenAIStructuredJudgeClient:
    """Request strict JSON-schema feedback through OpenAI's Responses API."""

    def __init__(
        self,
        *,
        model: str,
        model_revision: str,
        prompt_schema_version: str,
        api_key_environment: str = "OPENAI_API_KEY",
        client: Any | None = None,
    ) -> None:
        values = (model, model_revision, prompt_schema_version, api_key_environment)
        if any(not value.strip() for value in values):
            raise ValueError("judge provider identity fields must not be blank")
        if client is None:
            api_key = os.environ.get(api_key_environment, "")
            if not api_key.strip():
                raise RuntimeError(f"required judge secret {api_key_environment!r} is absent")
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("the OpenAI package is required for the API judge") from exc
            client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.model_revision = model_revision
        self.prompt_schema_version = prompt_schema_version
        self.client = client
        self.calls: list[JudgeProviderCall] = []

    async def complete(self, request: StructuredJudgeRequest) -> str:
        """Submit private context only to the provider and return validated JSON text."""
        request_payload = request.to_payload()
        request_json = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        request_fingerprint = hashlib.sha256(request_json.encode()).hexdigest()
        started = time.perf_counter()
        response = await self.client.responses.create(
            model=self.model,
            instructions=request.instructions,
            input=request_json,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "trajectory_feedback",
                    "schema": request.response_schema,
                    "strict": True,
                }
            },
        )
        latency = time.perf_counter() - started
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("OpenAI judge returned no structured output text")
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        self.calls.append(
            JudgeProviderCall(
                provider="openai",
                model=self.model,
                model_revision=self.model_revision,
                prompt_schema_version=self.prompt_schema_version,
                request_fingerprint=request_fingerprint,
                latency_seconds=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        return output_text


class DeterministicFakeStructuredJudgeClient:
    """Create schema-valid, edge-local feedback without network access."""

    def __init__(
        self,
        *,
        trajectory_score: float = 0.5,
        diagnostic: str = "The question is relevant; verify the local inference before relying on it.",
    ) -> None:
        if not 0.0 <= trajectory_score <= 1.0:
            raise ValueError("fake trajectory score must be in [0, 1]")
        if not diagnostic.strip():
            raise ValueError("fake diagnostic must not be blank")
        self.trajectory_score = trajectory_score
        self.diagnostic = diagnostic
        self.calls: list[JudgeProviderCall] = []

    async def complete(self, request: StructuredJudgeRequest) -> dict[str, Any]:
        """Derive deterministic feedback from public node topology only."""
        started = time.perf_counter()
        nodes = list(request.trajectory.get("nodes") or [])
        subcalls: list[InformationValueFeedback] = []
        for node in nodes:
            if node.get("kind") != "subcall":
                continue
            parent_id = str(node.get("parent_id") or "")
            child_id = str(node.get("node_id") or "")
            if not parent_id or not child_id:
                raise ValueError("fake judge received malformed subcall topology")
            subcalls.append(
                InformationValueFeedback(
                    parent_node_id=parent_id,
                    child_node_id=child_id,
                    information_significance=0.5,
                    novelty=0.5,
                    uncertainty_reduction=0.5,
                    evidence_quality=0.75,
                    edge_local_diagnostic=self.diagnostic,
                    rationale="deterministic fake assessment",
                )
            )
        outcome = (
            "correct"
            if self.trajectory_score == 1.0
            else "incorrect"
            if self.trajectory_score == 0.0
            else "partial"
        )
        feedback = TrajectoryFeedback(
            trajectory_score=self.trajectory_score,
            final_answer_feedback=FinalAnswerFeedback(outcome=outcome),
            subcalls=subcalls,
            judge_version=request.judge_version,
            rubric_version=request.rubric_version,
            metadata={"provider": "deterministic-fake-v1"},
        )
        safe_payload = {
            "task": request.task,
            "trajectory": request.trajectory,
            "judge_version": request.judge_version,
            "rubric_version": request.rubric_version,
        }
        encoded = json.dumps(safe_payload, sort_keys=True, separators=(",", ":")).encode()
        self.calls.append(
            JudgeProviderCall(
                provider="fake",
                model="deterministic-fake",
                model_revision="v1",
                prompt_schema_version="trajectory-feedback-v1",
                request_fingerprint=hashlib.sha256(encoded).hexdigest(),
                latency_seconds=time.perf_counter() - started,
                input_tokens=None,
                output_tokens=None,
            )
        )
        return feedback.model_dump(mode="json")


def _usage_value(usage: Any, name: str) -> int | None:
    """Read SDK object or mapping token usage without depending on one SDK release."""
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return int(value) if value is not None else None


__all__ = [
    "DeterministicFakeStructuredJudgeClient",
    "JudgeProviderCall",
    "OpenAIStructuredJudgeClient",
]
