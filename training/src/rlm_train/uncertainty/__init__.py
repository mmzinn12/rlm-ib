"""Semantic uncertainty domain package."""

from rlm_train.uncertainty.protocols import (
    AnswerSampler,
    SemanticEntropyEstimator,
    SemanticEquivalenceClassifier,
)
from rlm_train.uncertainty.schema import (
    AnswerSamplingRequest,
    SemanticCluster,
    SemanticEntropyEstimate,
    SemanticSample,
    SemanticSampleBatch,
    UncertaintyReduction,
)
from rlm_train.uncertainty.semantic_entropy import (
    ProbabilityWeightedSemanticEntropyEstimator,
    calculate_uncertainty_reduction,
    jensen_shannon_divergence,
    shannon_entropy,
)
from rlm_train.uncertainty.semantic_equivalence import (
    ExactMatchEquivalenceClassifier,
    TransformersNLIEquivalenceClassifier,
    cluster_semantic_samples,
)

__all__ = [
    "AnswerSampler",
    "AnswerSamplingRequest",
    "ExactMatchEquivalenceClassifier",
    "ProbabilityWeightedSemanticEntropyEstimator",
    "SemanticCluster",
    "SemanticEntropyEstimate",
    "SemanticEntropyEstimator",
    "SemanticEquivalenceClassifier",
    "SemanticSample",
    "SemanticSampleBatch",
    "TransformersNLIEquivalenceClassifier",
    "UncertaintyReduction",
    "calculate_uncertainty_reduction",
    "cluster_semantic_samples",
    "jensen_shannon_divergence",
    "shannon_entropy",
]
