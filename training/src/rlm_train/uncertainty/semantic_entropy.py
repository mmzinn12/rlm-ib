"""Pure probability-weighted semantic entropy calculations."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from rlm_train.uncertainty.schema import (
    SemanticCluster,
    SemanticEntropyEstimate,
    SemanticSampleBatch,
    UncertaintyReduction,
)

ESTIMATOR_NAME = "semantic_entropy"
ESTIMATOR_VERSION = "semantic-entropy-v1-natural-log"


class ProbabilityWeightedSemanticEntropyEstimator:
    def __init__(self, *, estimator_version: str = ESTIMATOR_VERSION) -> None:
        self.estimator_version = estimator_version
        self.cache: dict[str, SemanticEntropyEstimate] = {}

    def estimate(
        self,
        batch: SemanticSampleBatch,
        clusters: tuple[SemanticCluster, ...],
    ) -> SemanticEntropyEstimate:
        cache_payload = {
            "batch": batch.fingerprint,
            "clusters": [cluster.fingerprint for cluster in clusters],
            "estimator_version": self.estimator_version,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        samples = batch.samples
        condition = batch.condition
        sample_ids = {sample.sample_id for sample in samples}
        clustered_members = [
            member
            for cluster in clusters
            for member in cluster.member_sample_ids
            if member in sample_ids
        ]
        if set(clustered_members) != sample_ids or len(clustered_members) != len(sample_ids):
            raise ValueError("clusters must cover every estimated sample exactly")
        log_masses = {
            cluster.cluster_id: cluster.condition_log_probability_mass[condition]
            for cluster in clusters
            if condition in cluster.condition_log_probability_mass
        }
        if not log_masses:
            raise ValueError("clusters contain no probability mass for the requested condition")
        normalization = _logsumexp(tuple(log_masses.values()))
        probabilities = {
            cluster_id: math.exp(log_mass - normalization)
            for cluster_id, log_mass in sorted(log_masses.items())
        }
        total = math.fsum(probabilities.values())
        probabilities = {key: value / total for key, value in probabilities.items()}
        entropy = shannon_entropy(tuple(probabilities.values()))
        estimate = SemanticEntropyEstimate(
            estimator_name=ESTIMATOR_NAME,
            estimator_version=self.estimator_version,
            condition=condition,
            entropy=entropy,
            sample_count=len(samples),
            cluster_count=len(probabilities),
            cluster_probability_distribution=probabilities,
            model_identity=batch.model_identity,
            tokenizer_identity=batch.tokenizer_identity,
            sampling_provenance=batch.sampling_parameters,
            prompt_provenance={
                "fingerprint": batch.prompt_fingerprint,
                "version": batch.prompt_version,
            },
        )
        self.cache[cache_key] = estimate
        return estimate


def shannon_entropy(probabilities: tuple[float, ...]) -> float:
    _validate_distribution(probabilities)
    return -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)


def jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = tuple(sorted(set(left) | set(right)))
    if not keys:
        raise ValueError("Jensen-Shannon divergence requires non-empty distributions")
    left_values = tuple(left.get(key, 0.0) for key in keys)
    right_values = tuple(right.get(key, 0.0) for key in keys)
    _validate_distribution(left_values)
    _validate_distribution(right_values)
    midpoint = tuple((a + b) / 2.0 for a, b in zip(left_values, right_values, strict=True))
    result = (_kl(left_values, midpoint) + _kl(right_values, midpoint)) / 2.0
    return min(math.log(2.0), max(0.0, result))


def calculate_uncertainty_reduction(
    *,
    rollout_id: str,
    edge_id: str,
    before: SemanticEntropyEstimate,
    after: SemanticEntropyEstimate,
    clusters: tuple[SemanticCluster, ...],
    equivalence_provenance: dict[str, Any],
    sampling_seconds: float = 0.0,
    equivalence_seconds: float = 0.0,
) -> UncertaintyReduction:
    delta = before.entropy - after.entropy
    normalized = None if before.entropy == 0.0 else delta / before.entropy
    partition = [cluster.model_dump(mode="json") for cluster in clusters]
    fingerprint = hashlib.sha256(
        json.dumps(partition, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return UncertaintyReduction(
        rollout_id=rollout_id,
        edge_id=edge_id,
        before=before,
        after=after,
        absolute_entropy_reduction=delta,
        normalized_entropy_reduction=normalized,
        semantic_distribution_shift=jensen_shannon_divergence(
            before.cluster_probability_distribution, after.cluster_probability_distribution
        ),
        shared_cluster_partition_fingerprint=fingerprint,
        equivalence_provenance=equivalence_provenance,
        sampling_seconds=sampling_seconds,
        equivalence_seconds=equivalence_seconds,
    )


def _validate_distribution(probabilities: tuple[float, ...]) -> None:
    if not probabilities or not all(
        math.isfinite(value) and value >= 0.0 for value in probabilities
    ):
        raise ValueError("probability distribution must be non-empty, finite, and non-negative")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("probability distribution must sum to one")


def _kl(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(a * math.log(a / b) for a, b in zip(left, right, strict=True) if a > 0.0)


def _logsumexp(values: tuple[float, ...]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("log probability masses must be non-empty and finite")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


__all__ = [
    "ESTIMATOR_NAME",
    "ESTIMATOR_VERSION",
    "ProbabilityWeightedSemanticEntropyEstimator",
    "calculate_uncertainty_reduction",
    "jensen_shannon_divergence",
    "shannon_entropy",
]
