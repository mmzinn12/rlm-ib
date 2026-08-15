"""Immutable records for semantic uncertainty measurements."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Condition = Literal["before", "after"]
PROBABILITY_TOLERANCE = 1e-9


class ImmutableUncertaintyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class AnswerSamplingRequest(ImmutableUncertaintyRecord):
    condition: Condition
    rollout_id: str = Field(min_length=1)
    edge_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    context: Any
    helper_information: str
    prompt: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    sample_count: int = Field(ge=2)
    seeds: tuple[int, ...]
    temperature: float = Field(gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_seeds(self) -> AnswerSamplingRequest:
        if len(self.seeds) != self.sample_count or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("sampling request requires one unique seed per sample")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("sampling seeds must be non-negative")
        return self


class SemanticSample(ImmutableUncertaintyRecord):
    sample_id: str = Field(min_length=1)
    answer: str
    continuation_token_ids: tuple[int, ...]
    token_log_probabilities: tuple[float, ...]
    sampling_seed: int = Field(ge=0)

    @property
    def sequence_log_probability(self) -> float:
        return math.fsum(self.token_log_probabilities)

    @model_validator(mode="after")
    def validate_probabilities(self) -> SemanticSample:
        if not self.continuation_token_ids:
            raise ValueError("semantic samples require continuation token IDs")
        if len(self.continuation_token_ids) != len(self.token_log_probabilities):
            raise ValueError("token IDs and token log probabilities must align")
        if not all(math.isfinite(value) for value in self.token_log_probabilities):
            raise ValueError("semantic sample log probabilities must be finite")
        return self


class SemanticSampleBatch(ImmutableUncertaintyRecord):
    """Samples sharing one condition, prompt, model, and sampling configuration."""

    condition: Condition
    samples: tuple[SemanticSample, ...]
    model_identity: str = Field(min_length=1)
    tokenizer_identity: str = Field(min_length=1)
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    prompt_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_samples(self) -> SemanticSampleBatch:
        if not self.samples:
            raise ValueError("semantic sample batches must not be empty")
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("semantic sample IDs must be unique within a batch")
        seeds = tuple(sample.sampling_seed for sample in self.samples)
        if len(set(seeds)) != len(seeds):
            raise ValueError("semantic sampling seeds must be unique within a batch")
        return self


class SemanticCluster(ImmutableUncertaintyRecord):
    cluster_id: str = Field(pattern=r"^semantic-[0-9a-f]{16}$")
    member_sample_ids: tuple[str, ...]
    representative_answer: str
    condition_log_probability_mass: dict[Condition, float]

    @model_validator(mode="after")
    def validate_cluster(self) -> SemanticCluster:
        if (
            not self.member_sample_ids
            or tuple(sorted(self.member_sample_ids)) != self.member_sample_ids
        ):
            raise ValueError("cluster member IDs must be non-empty, unique, and sorted")
        if len(set(self.member_sample_ids)) != len(self.member_sample_ids):
            raise ValueError("cluster member IDs must be unique")
        if set(self.condition_log_probability_mass) - {"before", "after"}:
            raise ValueError("cluster probability mass has an unknown condition")
        if not all(math.isfinite(value) for value in self.condition_log_probability_mass.values()):
            raise ValueError("cluster log probability masses must be finite")
        return self


class SemanticEntropyEstimate(ImmutableUncertaintyRecord):
    estimator_name: str = Field(min_length=1)
    estimator_version: str = Field(min_length=1)
    logarithm_base: str = "e"
    condition: Condition
    entropy: float
    sample_count: int = Field(gt=0)
    cluster_count: int = Field(gt=0)
    cluster_probability_distribution: dict[str, float]
    model_identity: str = Field(min_length=1)
    tokenizer_identity: str = Field(min_length=1)
    sampling_provenance: dict[str, Any]
    prompt_provenance: dict[str, Any]

    @model_validator(mode="after")
    def validate_distribution(self) -> SemanticEntropyEstimate:
        if not math.isfinite(self.entropy) or self.entropy < 0.0:
            raise ValueError("semantic entropy must be finite and non-negative")
        if len(self.cluster_probability_distribution) != self.cluster_count:
            raise ValueError("cluster count must match the probability distribution")
        probabilities = tuple(self.cluster_probability_distribution.values())
        if not probabilities or not all(
            math.isfinite(value) and value >= 0.0 for value in probabilities
        ):
            raise ValueError("cluster probabilities must be finite and non-negative")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=PROBABILITY_TOLERANCE):
            raise ValueError("cluster probabilities must sum to one")
        return self


class UncertaintyReduction(ImmutableUncertaintyRecord):
    rollout_id: str = Field(min_length=1)
    edge_id: str = Field(min_length=1)
    before: SemanticEntropyEstimate
    after: SemanticEntropyEstimate
    absolute_entropy_reduction: float
    normalized_entropy_reduction: float | None
    semantic_distribution_shift: float
    shared_cluster_partition_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    equivalence_provenance: dict[str, Any]
    sampling_seconds: float = Field(default=0.0, ge=0.0)
    equivalence_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_reduction(self) -> UncertaintyReduction:
        if self.before.condition != "before" or self.after.condition != "after":
            raise ValueError("uncertainty reduction requires before and after estimates")
        numbers = (self.absolute_entropy_reduction, self.semantic_distribution_shift)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("uncertainty reduction values must be finite")
        if self.normalized_entropy_reduction is not None and not math.isfinite(
            self.normalized_entropy_reduction
        ):
            raise ValueError("normalized entropy reduction must be finite when defined")
        if not 0.0 <= self.semantic_distribution_shift <= math.log(2.0) + 1e-12:
            raise ValueError("natural-log Jensen-Shannon divergence must lie in [0, ln(2)]")
        return self


__all__ = [
    "AnswerSamplingRequest",
    "Condition",
    "PROBABILITY_TOLERANCE",
    "SemanticCluster",
    "SemanticEntropyEstimate",
    "SemanticSample",
    "SemanticSampleBatch",
    "UncertaintyReduction",
]
