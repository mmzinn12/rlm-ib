"""Run deterministic, resumable benchmark evaluation and aggregate acc/pass reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from rlm.core.trajectory import TrajectoryTree

from rlm_train.benchmarks.types import (
    Benchmark,
    BenchmarkIdentity,
    BenchmarkRole,
    EvaluationRecord,
    ModelProvenance,
)
from rlm_train.diagnostics import collect_observer_diagnostics


@dataclass(frozen=True)
class GenerationResult:
    """Return sampled text and optional detached observations from a generator."""

    response: str
    response_tokens: tuple[str, ...] | None = None
    token_count: int | None = None
    truncated: bool = False
    trajectory: TrajectoryTree | None = None
    per_token_divergence: tuple[float, ...] | None = None
    gram_metrics: Mapping[str, Any] | None = None


class ResponseGenerator(Protocol):
    """Generate one response using public problem data and a deterministic seed."""

    async def generate(
        self,
        *,
        prompt: str,
        public_problem: dict[str, Any],
        seed: int,
        sample_index: int,
        model: ModelProvenance,
    ) -> GenerationResult:
        """Sample a response without receiving verifier-owned target data."""
        ...


class EvaluationAggregate(BaseModel):
    """Store prefix metrics for every k from one through samples-per-problem."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    acc_at_k: dict[str, float]
    pass_at_k: dict[str, float]
    mean_response_tokens: float = Field(ge=0.0)
    truncation_rate: float = Field(ge=0.0, le=1.0)


class EvaluationReport(BaseModel):
    """Persist aggregate results together with exact dataset and run configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkIdentity
    model: ModelProvenance
    samples_per_problem: int = Field(gt=0)
    configuration: dict[str, Any]
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregate: EvaluationAggregate
    generated_record_count: int = Field(ge=0)
    resumed_record_count: int = Field(ge=0)


class JSONLEvaluationStore:
    """Append unique evaluation records and use them as a durable resume journal."""

    def __init__(self, path: str | Path) -> None:
        if not str(path).strip():
            raise ValueError("evaluation store path must not be blank")
        self.path = Path(path)

    def append(self, record: EvaluationRecord) -> None:
        """Persist one record after rejecting a duplicate deterministic key."""
        if record.record_key in self.records_by_key():
            raise ValueError(f"duplicate evaluation record {record.record_key}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{line}\n")

    def iter_records(self) -> Iterator[EvaluationRecord]:
        """Yield validated records and identify corrupt source lines."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield EvaluationRecord.model_validate_json(line)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid evaluation record on line {line_number}") from exc

    def records_by_key(self) -> dict[str, EvaluationRecord]:
        """Return the latest validated resume index (keys are unique by construction)."""
        records: dict[str, EvaluationRecord] = {}
        for record in self.iter_records():
            if record.record_key in records:
                raise ValueError(f"duplicate evaluation record {record.record_key} in store")
            records[record.record_key] = record
        return records


class BenchmarkEvaluator:
    """Evaluate any benchmark protocol without importing a trainer or dataset SDK."""

    def __init__(
        self,
        store: JSONLEvaluationStore,
        *,
        base_seed: int,
        allowed_lockbox_checkpoint_steps: Sequence[int] = (),
    ) -> None:
        if base_seed < 0:
            raise ValueError("evaluation base_seed must be non-negative")
        if any(step < 0 for step in allowed_lockbox_checkpoint_steps):
            raise ValueError("lockbox checkpoint steps must be non-negative")
        if len(allowed_lockbox_checkpoint_steps) != len(set(allowed_lockbox_checkpoint_steps)):
            raise ValueError("lockbox checkpoint steps must be unique")
        self.store = store
        self.base_seed = base_seed
        self.allowed_lockbox_checkpoint_steps = tuple(allowed_lockbox_checkpoint_steps)

    async def evaluate(
        self,
        benchmark: Benchmark,
        generator: ResponseGenerator,
        *,
        model: ModelProvenance,
        samples_per_problem: int,
        configuration: Mapping[str, Any],
        diagnostics_enabled: bool = True,
        diagnostics_configuration: Mapping[str, bool] | None = None,
    ) -> EvaluationReport:
        """Generate only missing samples, persist each immediately, and aggregate."""
        if samples_per_problem <= 0:
            raise ValueError("samples_per_problem must be positive")
        if (
            benchmark.identity.role is BenchmarkRole.LOCKBOX
            and model.checkpoint_step not in self.allowed_lockbox_checkpoint_steps
        ):
            raise ValueError("lockbox evaluation is not scheduled for this checkpoint step")
        problems = tuple(benchmark.problems())
        if not problems:
            raise ValueError("benchmark must contain at least one problem")
        existing = self.store.records_by_key()
        selected: list[EvaluationRecord] = []
        generated_count = 0
        resumed_count = 0
        for problem in problems:
            prompt = benchmark.format_prompt(problem)
            if problem.target_data is None:
                raise ValueError("benchmark problems require verifier-owned target data")
            for sample_index in range(samples_per_problem):
                key = _evaluation_key(benchmark.identity, problem.problem_id, sample_index, model)
                cached = existing.get(key)
                if cached is not None:
                    selected.append(cached)
                    resumed_count += 1
                    continue
                seed = derive_evaluation_seed(
                    self.base_seed,
                    benchmark.identity,
                    problem.problem_id,
                    sample_index,
                )
                result = await generator.generate(
                    prompt=prompt,
                    public_problem=problem.public_payload(),
                    seed=seed,
                    sample_index=sample_index,
                    model=model,
                )
                extraction = benchmark.extract_answer(result.response)
                score = benchmark.score(problem, extraction)
                diagnostics = {}
                if diagnostics_enabled:
                    observed = collect_observer_diagnostics(
                        result.response,
                        response_tokens=result.response_tokens,
                        token_count=result.token_count,
                        truncated=result.truncated,
                        trajectory=result.trajectory,
                        per_token_divergence=(
                            result.per_token_divergence
                            if _diagnostic_enabled(diagnostics_configuration, "divergence")
                            else None
                        ),
                        gram_metrics=(
                            result.gram_metrics
                            if _diagnostic_enabled(diagnostics_configuration, "gram_drift")
                            else None
                        ),
                    ).model_dump(mode="json")
                    diagnostics = {
                        "response_token_count": observed["response_token_count"],
                        "truncated": observed["truncated"],
                    }
                    if _diagnostic_enabled(diagnostics_configuration, "epistemic_markers"):
                        diagnostics["epistemic"] = observed["epistemic"]
                    if _diagnostic_enabled(diagnostics_configuration, "reasoning_dynamics"):
                        diagnostics["reasoning"] = observed["reasoning"]
                        diagnostics["trajectory"] = observed["trajectory"]
                    if _diagnostic_enabled(diagnostics_configuration, "divergence"):
                        diagnostics["divergence"] = observed["divergence"]
                    if _diagnostic_enabled(diagnostics_configuration, "gram_drift"):
                        diagnostics["gram"] = observed["gram"]
                record = EvaluationRecord(
                    benchmark=benchmark.identity,
                    problem_id=problem.problem_id,
                    public_prompt=prompt,
                    response=result.response,
                    extraction=extraction,
                    score=score,
                    diagnostics=diagnostics,
                    seed=seed,
                    sample_index=sample_index,
                    model=model,
                )
                if record.record_key != key:
                    raise RuntimeError("evaluation record key construction is inconsistent")
                self.store.append(record)
                existing[key] = record
                selected.append(record)
                generated_count += 1
        resolved_configuration = json.loads(
            json.dumps(configuration, sort_keys=True, allow_nan=False)
        )
        config_json = json.dumps(
            resolved_configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return EvaluationReport(
            benchmark=benchmark.identity,
            model=model,
            samples_per_problem=samples_per_problem,
            configuration=resolved_configuration,
            configuration_fingerprint=hashlib.sha256(config_json.encode()).hexdigest(),
            aggregate=aggregate_evaluation_records(
                selected,
                problem_ids=tuple(problem.problem_id for problem in problems),
                samples_per_problem=samples_per_problem,
            ),
            generated_record_count=generated_count,
            resumed_record_count=resumed_count,
        )


def derive_evaluation_seed(
    base_seed: int,
    identity: BenchmarkIdentity,
    problem_id: str,
    sample_index: int,
) -> int:
    """Derive a stable 63-bit seed independent of process ordering."""
    if base_seed < 0 or sample_index < 0:
        raise ValueError("evaluation seed inputs must be non-negative")
    payload = f"{base_seed}\0{identity.key}\0{problem_id}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def aggregate_evaluation_records(
    records: Sequence[EvaluationRecord],
    *,
    problem_ids: Sequence[str],
    samples_per_problem: int,
) -> EvaluationAggregate:
    """Compute observed cumulative accuracy and any-success pass rate for every k."""
    grouped: dict[str, dict[int, EvaluationRecord]] = {problem_id: {} for problem_id in problem_ids}
    for record in records:
        if record.problem_id not in grouped:
            raise ValueError("evaluation record references an unexpected problem")
        if record.sample_index >= samples_per_problem:
            raise ValueError("evaluation record sample index exceeds report configuration")
        if record.sample_index in grouped[record.problem_id]:
            raise ValueError("evaluation report contains a duplicate problem/sample pair")
        grouped[record.problem_id][record.sample_index] = record
    for problem_id, samples in grouped.items():
        missing = set(range(samples_per_problem)) - set(samples)
        if missing:
            raise ValueError(f"evaluation report is missing samples for {problem_id!r}: {missing}")
    acc_at_k: dict[str, float] = {}
    pass_at_k: dict[str, float] = {}
    for k in range(1, samples_per_problem + 1):
        correctness = [
            grouped[problem_id][sample_index].score.correct
            for problem_id in problem_ids
            for sample_index in range(k)
        ]
        acc_at_k[str(k)] = sum(correctness) / len(correctness)
        pass_at_k[str(k)] = sum(
            any(grouped[problem_id][sample_index].score.correct for sample_index in range(k))
            for problem_id in problem_ids
        ) / len(problem_ids)
    token_counts = [
        int(record.diagnostics.get("response_token_count", len(record.response.split())))
        for record in records
    ]
    truncations = [bool(record.diagnostics.get("truncated", False)) for record in records]
    return EvaluationAggregate(
        problem_count=len(problem_ids),
        sample_count=len(records),
        acc_at_k=acc_at_k,
        pass_at_k=pass_at_k,
        mean_response_tokens=sum(token_counts) / len(token_counts),
        truncation_rate=sum(truncations) / len(truncations),
    )


def _evaluation_key(
    identity: BenchmarkIdentity,
    problem_id: str,
    sample_index: int,
    model: ModelProvenance,
) -> str:
    """Mirror ``EvaluationRecord.record_key`` before a response exists."""
    payload = {
        "benchmark": identity.key,
        "problem_id": problem_id,
        "sample_index": sample_index,
        "model": model.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _diagnostic_enabled(configuration: Mapping[str, bool] | None, name: str) -> bool:
    """Resolve one observer flag, defaulting to enabled when no mapping is supplied."""
    return True if configuration is None else bool(configuration.get(name, False))


__all__ = [
    "BenchmarkEvaluator",
    "EvaluationAggregate",
    "EvaluationReport",
    "GenerationResult",
    "JSONLEvaluationStore",
    "ResponseGenerator",
    "aggregate_evaluation_records",
    "derive_evaluation_seed",
]
