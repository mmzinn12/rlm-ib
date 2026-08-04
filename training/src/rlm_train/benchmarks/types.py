"""Define the framework-neutral benchmark and evaluation record contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableRecord(BaseModel):
    """Reject unknown fields and prevent evaluation provenance from drifting."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkRole(StrEnum):
    """Separate development benchmarks from predetermined lockbox evaluations."""

    DEVELOPMENT = "development"
    LOCKBOX = "lockbox"


class ExtractionStatus(StrEnum):
    """Distinguish successful extraction from empty and malformed responses."""

    EXTRACTED = "extracted"
    MISSING = "missing"
    MALFORMED = "malformed"


class BenchmarkIdentity(ImmutableRecord):
    """Identify an exact benchmark snapshot and its experimental role."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: BenchmarkRole = BenchmarkRole.DEVELOPMENT

    @property
    def key(self) -> str:
        """Return a compact stable identity used by stores and reports."""
        return f"{self.name}:{self.version}:{self.split}:{self.source_fingerprint}"


class Problem(ImmutableRecord):
    """Carry public prompt data and verifier-owned target data separately."""

    problem_id: str = Field(min_length=1)
    public_prompt: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    target_data: Any = Field(repr=False)

    def public_payload(self) -> dict[str, Any]:
        """Return the problem fields safe to expose to a response generator."""
        return {
            "problem_id": self.problem_id,
            "public_prompt": self.public_prompt,
            "metadata": json.loads(json.dumps(self.metadata)),
        }


class ExtractedAnswer(ImmutableRecord):
    """Record normalized answer text and why extraction did or did not succeed."""

    normalized_answer: str | None = None
    status: ExtractionStatus

    @model_validator(mode="after")
    def validate_status(self) -> ExtractedAnswer:
        """Require text exactly when extraction succeeded."""
        if self.status is ExtractionStatus.EXTRACTED:
            if self.normalized_answer is None or not self.normalized_answer:
                raise ValueError("extracted answers require normalized text")
        elif self.normalized_answer is not None:
            raise ValueError("missing or malformed answers cannot carry normalized text")
        return self


class Score(ImmutableRecord):
    """Store a numeric reward, correctness, and structured failure reason."""

    reward: float = Field(ge=0.0, le=1.0)
    correct: bool
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> Score:
        """Keep correctness and failure reason mutually consistent."""
        if self.correct and self.failure_reason is not None:
            raise ValueError("correct scores cannot have a failure reason")
        if not self.correct and (self.failure_reason is None or not self.failure_reason.strip()):
            raise ValueError("incorrect scores require a failure reason")
        return self


class ModelProvenance(ImmutableRecord):
    """Identify the model, checkpoint, prompt, and generation policy evaluated."""

    model_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    checkpoint_step: int = Field(ge=0)
    prompt_version: str = Field(min_length=1)
    generation_parameters: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(ImmutableRecord):
    """Persist one response and all observer-only measurements needed for replay."""

    benchmark: BenchmarkIdentity
    problem_id: str = Field(min_length=1)
    public_prompt: str = Field(min_length=1)
    response: str
    extraction: ExtractedAnswer
    score: Score
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(ge=0)
    sample_index: int = Field(ge=0)
    model: ModelProvenance

    @property
    def record_key(self) -> str:
        """Return the deterministic resume key for this exact evaluation sample."""
        payload = {
            "benchmark": self.benchmark.key,
            "problem_id": self.problem_id,
            "sample_index": self.sample_index,
            "model": self.model.model_dump(mode="json"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class Benchmark(Protocol):
    """Define the only benchmark surface consumed by the evaluator."""

    @property
    def identity(self) -> BenchmarkIdentity:
        """Return the exact dataset identity and role."""
        ...

    def problems(self) -> Sequence[Problem]:
        """Return problems in deterministic evaluation order."""
        ...

    def format_prompt(self, problem: Problem) -> str:
        """Build the public model prompt for one problem."""
        ...

    def extract_answer(self, response: str) -> ExtractedAnswer:
        """Extract and normalize an answer from a sampled response."""
        ...

    def score(self, problem: Problem, answer: ExtractedAnswer) -> Score:
        """Verify an extracted answer against target data held by the adapter."""
        ...


__all__ = [
    "Benchmark",
    "BenchmarkIdentity",
    "BenchmarkRole",
    "EvaluationRecord",
    "ExtractedAnswer",
    "ExtractionStatus",
    "ModelProvenance",
    "Problem",
    "Score",
]
