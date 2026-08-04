"""Launch the standalone single-GPU GRPO path used by Google Colab."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from rlm_train.benchmarks import (
    BenchmarkEvaluator,
    JSONLEvaluationStore,
    ModelProvenance,
    default_benchmark_registry,
)
from rlm_train.colab.checkpoint import TrainingCheckpointManager
from rlm_train.colab.config import ColabRunConfig
from rlm_train.colab.generation import TransformersResponseGenerator
from rlm_train.colab.runtime import (
    load_policy_bundle,
    validate_colab_runtime,
)
from rlm_train.colab.trainer import (
    MetricsJournal,
    SingleGPUTrainer,
    SmokeIndexRubric,
    load_training_dataset,
    trainable_parameter_fingerprint,
)
from rlm_train.experiment.config import TrainingAlgorithm


async def run_colab_training(
    configuration_path: str | Path,
    *,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Preflight, load LoRA, train, checkpoint, and run scheduled development evals."""
    config_path = Path(configuration_path).resolve()
    configuration = ColabRunConfig.from_file(config_path)
    preflight = validate_colab_runtime(
        configuration,
        base_directory=Path.cwd(),
    )
    if configuration.resolved_experiment.training.algorithm is not TrainingAlgorithm.GRPO:
        raise RuntimeError(
            "the command-line dataset adapter currently launches the policy-only GRPO arm; "
            "use SingleGPUTrainer with MaskedQuestionSDPOLossBuilder for traced SDPO batches"
        )
    run_directory = Path(preflight.output_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    bundle = load_policy_bundle(configuration, preflight)
    generator = TransformersResponseGenerator(
        bundle.model,
        bundle.tokenizer,
        configuration.generation,
        model_context_length=configuration.model.max_context_length,
    )
    registry = default_benchmark_registry()
    evaluation_benchmarks = []
    for specification in configuration.resolved_experiment.evaluation.benchmarks:
        evaluation_benchmarks.append(
            registry.create(
                specification.adapter,
                specification.factory_configuration(base_directory=Path.cwd()),
            )
        )
    training_dataset, overlap_results = load_training_dataset(
        configuration,
        base_directory=Path.cwd(),
        evaluation_benchmarks=evaluation_benchmarks,
    )
    run_provenance = {
        "configuration": configuration.resolved_dict(base_directory=Path.cwd()),
        "configuration_fingerprint": configuration.fingerprint(base_directory=Path.cwd()),
        "runtime": {
            "cuda_device_name": preflight.cuda_device_name,
            "precision": preflight.precision,
            "output_directory": preflight.output_directory,
            "dependency_versions": preflight.dependency_versions,
            "python_version": preflight.python_version,
        },
        "model": bundle.provenance,
        "dataset_fingerprint": training_dataset.identity.source_fingerprint,
        "benchmark_fingerprints": {
            benchmark.identity.key: benchmark.identity.source_fingerprint
            for benchmark in evaluation_benchmarks
        },
        "overlap_check_results": overlap_results,
        "source_revision": source_revision(),
    }
    _initialize_run_provenance(run_directory / "run-provenance.json", run_provenance)
    trainer = SingleGPUTrainer(
        model=bundle.model,
        generator=generator,
        training_dataset=training_dataset,
        configuration=configuration,
        rubric=(
            SmokeIndexRubric(configuration.generation.rollouts_per_prompt)
            if configuration.dataset.rubric == "smoke_index"
            else None
        ),
        metrics_journal=MetricsJournal(run_directory / "metrics.jsonl"),
    )
    checkpoint_manager = TrainingCheckpointManager(
        run_directory,
        configuration,
        base_directory=Path.cwd(),
    )
    initial_trainable_fingerprint = trainable_parameter_fingerprint(trainer.trainable_parameters)
    benchmark_fingerprints = {
        benchmark.identity.key: benchmark.identity.source_fingerprint
        for benchmark in evaluation_benchmarks
    }
    if resume is None and (run_directory / "latest.json").exists():
        raise RuntimeError("run already has checkpoints; pass --resume to continue it")
    if resume is not None:
        checkpoint_manager.restore(
            trainer,
            None if str(resume) == "latest" else resume,
            expected_model_identity=bundle.provenance,
            expected_tokenizer_fingerprint=bundle.tokenizer_fingerprint,
            expected_dataset_fingerprint=training_dataset.identity.source_fingerprint,
            expected_benchmark_fingerprints=benchmark_fingerprints,
        )

    async def checkpoint_and_evaluate(metrics: Any, active_trainer: SingleGPUTrainer) -> None:
        step = metrics.global_step
        if step % configuration.output.evaluate_every_steps == 0:
            await evaluate_development_benchmarks(
                configuration,
                generator,
                evaluation_benchmarks,
                bundle.provenance,
                run_directory,
                checkpoint_step=step,
            )
        if step % configuration.output.checkpoint_every_steps == 0:
            completed = completed_evaluation_keys(run_directory)
            checkpoint_manager.save(
                active_trainer,
                model_identity=bundle.provenance,
                tokenizer_fingerprint=bundle.tokenizer_fingerprint,
                dataset_fingerprint=training_dataset.identity.source_fingerprint,
                benchmark_fingerprints=benchmark_fingerprints,
                overlap_check_results=overlap_results,
                completed_evaluation_keys=completed,
                source_revision=run_provenance["source_revision"],
            )

    reports = await trainer.run(on_optimizer_step=checkpoint_and_evaluate)
    final_trainable_fingerprint = trainable_parameter_fingerprint(trainer.trainable_parameters)
    if (
        configuration.profile.value == "smoke"
        and final_trainable_fingerprint == initial_trainable_fingerprint
    ):
        raise RuntimeError("smoke optimizer step did not change a trainable student parameter")
    if trainer.state.global_step % configuration.output.checkpoint_every_steps != 0:
        checkpoint_manager.save(
            trainer,
            model_identity=bundle.provenance,
            tokenizer_fingerprint=bundle.tokenizer_fingerprint,
            dataset_fingerprint=training_dataset.identity.source_fingerprint,
            benchmark_fingerprints=benchmark_fingerprints,
            overlap_check_results=overlap_results,
            completed_evaluation_keys=completed_evaluation_keys(run_directory),
            source_revision=run_provenance["source_revision"],
        )
    summary = {
        "global_step": trainer.state.global_step,
        "checkpoint": str(checkpoint_manager.resolve_checkpoint()),
        "last_metrics": reports[-1].to_dict() if reports else None,
        "run_directory": str(run_directory),
        "initial_trainable_fingerprint": initial_trainable_fingerprint,
        "final_trainable_fingerprint": final_trainable_fingerprint,
    }
    _atomic_json(run_directory / "summary.json", summary)
    return summary


async def evaluate_development_benchmarks(
    configuration: ColabRunConfig,
    generator: TransformersResponseGenerator,
    benchmarks: list[Any],
    model_identity: dict[str, Any],
    run_directory: Path,
    *,
    checkpoint_step: int,
) -> None:
    """Run the existing generic evaluator against a real policy checkpoint."""
    evaluation = configuration.resolved_experiment.evaluation
    for benchmark in benchmarks:
        model = ModelProvenance(
            model_id=str(model_identity["model_id"]),
            checkpoint_id=f"checkpoint-{checkpoint_step:08d}",
            checkpoint_step=checkpoint_step,
            prompt_version=configuration.generation.prompt_template_version,
            generation_parameters=configuration.generation.model_dump(mode="json"),
        )
        evaluator = BenchmarkEvaluator(
            JSONLEvaluationStore(
                run_directory / "evaluation" / benchmark.identity.name / "records.jsonl"
            ),
            base_seed=evaluation.base_seed,
            allowed_lockbox_checkpoint_steps=evaluation.checkpoint_steps,
        )
        report = await evaluator.evaluate(
            benchmark,
            generator,
            model=model,
            samples_per_problem=evaluation.samples_per_problem,
            configuration=configuration.resolved_dict(base_directory=Path.cwd()),
            diagnostics_enabled=True,
            diagnostics_configuration=evaluation.diagnostics.model_dump(),
        )
        report_path = (
            run_directory
            / "evaluation"
            / benchmark.identity.name
            / f"report-{checkpoint_step:08d}.json"
        )
        _atomic_json(report_path, report.model_dump(mode="json"))


def completed_evaluation_keys(run_directory: Path) -> tuple[str, ...]:
    """Collect durable evaluation keys across generic JSONL stores."""
    keys: list[str] = []
    for path in sorted((run_directory / "evaluation").glob("*/records.jsonl")):
        store = JSONLEvaluationStore(path)
        keys.extend(store.records_by_key())
    return tuple(sorted(set(keys)))


def source_revision() -> str:
    """Resolve the exact Git source revision for reports and checkpoints."""
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = process.stdout.strip()
    if len(revision) != 40:
        raise RuntimeError("Git did not return a full source revision")
    return revision


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        f"{json.dumps(payload, sort_keys=True, allow_nan=False)}\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _initialize_run_provenance(path: Path, payload: dict[str, Any]) -> None:
    """Create or byte-validate immutable run-start provenance."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing run provenance differs from the requested run")
        return
    _atomic_json(path, payload)


def main() -> None:
    """Launch from a fresh Colab runtime after installing the optional dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configuration",
        nargs="?",
        default="training/configs/colab-smoke.toml",
    )
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    arguments = parser.parse_args()
    summary = asyncio.run(run_colab_training(arguments.configuration, resume=arguments.resume))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
