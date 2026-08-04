"""Expose generic, swappable, download-free benchmark evaluation APIs."""

from rlm_train.benchmarks.evaluator import (
    BenchmarkEvaluator,
    EvaluationAggregate,
    EvaluationReport,
    GenerationResult,
    JSONLEvaluationStore,
    ResponseGenerator,
    aggregate_evaluation_records,
    derive_evaluation_seed,
)
from rlm_train.benchmarks.jsonl import JSONLBenchmark, find_prompt_overlaps
from rlm_train.benchmarks.registry import BenchmarkRegistry, default_benchmark_registry
from rlm_train.benchmarks.types import (
    Benchmark,
    BenchmarkIdentity,
    BenchmarkRole,
    EvaluationRecord,
    ExtractedAnswer,
    ExtractionStatus,
    ModelProvenance,
    Problem,
    Score,
)

__all__ = [
    "Benchmark",
    "BenchmarkEvaluator",
    "BenchmarkIdentity",
    "BenchmarkRegistry",
    "BenchmarkRole",
    "EvaluationAggregate",
    "EvaluationRecord",
    "EvaluationReport",
    "ExtractedAnswer",
    "ExtractionStatus",
    "GenerationResult",
    "JSONLBenchmark",
    "JSONLEvaluationStore",
    "ModelProvenance",
    "Problem",
    "ResponseGenerator",
    "Score",
    "aggregate_evaluation_records",
    "default_benchmark_registry",
    "derive_evaluation_seed",
    "find_prompt_overlaps",
]
