"""Injectable contracts for semantic uncertainty implementations."""

from typing import Protocol

from rlm_train.uncertainty.schema import (
    AnswerSamplingRequest,
    SemanticCluster,
    SemanticEntropyEstimate,
    SemanticSampleBatch,
)


class AnswerSampler(Protocol):
    def sample(self, request: AnswerSamplingRequest) -> SemanticSampleBatch: ...


class SemanticEquivalenceClassifier(Protocol):
    @property
    def provenance(self) -> dict[str, str]: ...

    def equivalent(self, question: str, left: str, right: str) -> bool: ...


class SemanticEntropyEstimator(Protocol):
    def estimate(
        self,
        batch: SemanticSampleBatch,
        clusters: tuple[SemanticCluster, ...],
    ) -> SemanticEntropyEstimate: ...


__all__ = ["AnswerSampler", "SemanticEntropyEstimator", "SemanticEquivalenceClassifier"]
