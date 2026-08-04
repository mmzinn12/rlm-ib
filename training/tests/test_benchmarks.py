"""Verify the generic JSONL adapter, registry, deterministic resume, and reports."""

import json

import pytest

from rlm_train.benchmarks import (
    BenchmarkEvaluator,
    BenchmarkRole,
    ExtractionStatus,
    GenerationResult,
    JSONLBenchmark,
    JSONLEvaluationStore,
    ModelProvenance,
    default_benchmark_registry,
    find_prompt_overlaps,
)


def write_benchmark(path, *, role=BenchmarkRole.DEVELOPMENT):
    rows = [
        {"id": "p1", "prompt": "Compute 1+1", "target": "2"},
        {"id": "p2", "prompt": "Compute 2+2", "target": "4"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return JSONLBenchmark(
        path,
        name="synthetic",
        version="v1",
        split="dev",
        role=role,
        answer_pattern=r"FINAL:\s*(?P<answer>\d+)",
    )


class MappingGenerator:
    """Return deterministic answers while proving targets never reach the generator."""

    def __init__(self, *, fail_after=None):
        self.fail_after = fail_after
        self.calls = []

    async def generate(self, **kwargs):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("synthetic interruption")
        assert "target" not in kwargs["public_problem"]
        self.calls.append(kwargs)
        answer = {"p1": "2", "p2": "4"}[kwargs["public_problem"]["problem_id"]]
        if kwargs["sample_index"] == 1 and kwargs["public_problem"]["problem_id"] == "p2":
            answer = "5"
        return GenerationResult(response=f"reasoning\nFINAL: {answer}", token_count=3)


def model(step=20):
    return ModelProvenance(
        model_id="model",
        checkpoint_id=f"checkpoint-{step}",
        checkpoint_step=step,
        prompt_version="v1",
        generation_parameters={"temperature": 0.7},
    )


def test_jsonl_extraction_scoring_fingerprint_registry_and_overlap(tmp_path):
    benchmark = write_benchmark(tmp_path / "benchmark.jsonl")
    problem = benchmark.problems()[0]

    assert benchmark.extract_answer("").status is ExtractionStatus.MISSING
    assert benchmark.extract_answer("no final marker").status is ExtractionStatus.MALFORMED
    assert benchmark.score(problem, benchmark.extract_answer("FINAL: 2")).correct
    incorrect = benchmark.score(problem, benchmark.extract_answer("FINAL: 7"))
    assert incorrect.failure_reason == "incorrect_answer"
    assert len(benchmark.identity.source_fingerprint) == 64
    assert find_prompt_overlaps(benchmark, ["  compute   1+1  "]) == {"p1": (0,)}
    assert default_benchmark_registry().adapter_names == ("jsonl",)


@pytest.mark.asyncio
async def test_evaluation_resumes_without_duplicate_samples_and_reports_acc_pass(tmp_path):
    benchmark = write_benchmark(tmp_path / "benchmark.jsonl")
    store = JSONLEvaluationStore(tmp_path / "records.jsonl")
    evaluator = BenchmarkEvaluator(store, base_seed=42)
    interrupted = MappingGenerator(fail_after=2)

    with pytest.raises(RuntimeError, match="interruption"):
        await evaluator.evaluate(
            benchmark,
            interrupted,
            model=model(),
            samples_per_problem=2,
            configuration={"run": "synthetic"},
        )
    assert len(tuple(store.iter_records())) == 2

    resumed = MappingGenerator()
    report = await evaluator.evaluate(
        benchmark,
        resumed,
        model=model(),
        samples_per_problem=2,
        configuration={"run": "synthetic"},
    )

    assert report.generated_record_count == 2
    assert report.resumed_record_count == 2
    assert len(tuple(store.iter_records())) == 4
    assert report.aggregate.acc_at_k == {"1": 1.0, "2": 0.75}
    assert report.aggregate.pass_at_k == {"1": 1.0, "2": 1.0}
    assert report.aggregate.mean_response_tokens == 3.0
    assert len({record.seed for record in store.iter_records()}) == 4


@pytest.mark.asyncio
async def test_lockbox_role_runs_only_at_predetermined_checkpoints(tmp_path):
    benchmark = write_benchmark(tmp_path / "lockbox.jsonl", role=BenchmarkRole.LOCKBOX)
    evaluator = BenchmarkEvaluator(
        JSONLEvaluationStore(tmp_path / "records.jsonl"),
        base_seed=0,
        allowed_lockbox_checkpoint_steps=(20,),
    )

    with pytest.raises(ValueError, match="not scheduled"):
        await evaluator.evaluate(
            benchmark,
            MappingGenerator(),
            model=model(step=19),
            samples_per_problem=1,
            configuration={},
        )


@pytest.mark.asyncio
async def test_diagnostic_flags_change_only_observer_payloads(tmp_path):
    benchmark = write_benchmark(tmp_path / "benchmark.jsonl")
    evaluator = BenchmarkEvaluator(
        JSONLEvaluationStore(tmp_path / "records.jsonl"),
        base_seed=7,
    )
    report = await evaluator.evaluate(
        benchmark,
        MappingGenerator(),
        model=model(),
        samples_per_problem=1,
        configuration={"diagnostics": "disabled"},
        diagnostics_configuration={
            "epistemic_markers": False,
            "reasoning_dynamics": False,
            "divergence": False,
            "gram_drift": False,
        },
    )

    records = tuple(evaluator.store.iter_records())
    assert report.aggregate.acc_at_k == {"1": 1.0}
    assert all(
        set(record.diagnostics) == {"response_token_count", "truncated"} for record in records
    )
