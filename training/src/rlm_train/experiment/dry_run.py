"""Execute the local synthetic benchmark path without a model, dataset, or Prime download."""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import re
from pathlib import Path
from typing import Any

from rlm_train.benchmarks import (
    BenchmarkEvaluator,
    GenerationResult,
    JSONLEvaluationStore,
    ModelProvenance,
    default_benchmark_registry,
)
from rlm_train.experiment.config import ExperimentConfig, TrainingAlgorithm
from rlm_train.experiment.lifecycle import RunArtifactStore

_ARITHMETIC_PATTERN = re.compile(r"Compute\s+(-?\d+)\s*([+*\-])\s*(-?\d+)", re.IGNORECASE)
_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


class SyntheticArithmeticGenerator:
    """Solve the public arithmetic prompt to exercise evaluation plumbing only."""

    async def generate(self, **kwargs: Any) -> GenerationResult:
        """Return deterministic public-prompt-derived text without inspecting targets."""
        prompt = str(kwargs["prompt"])
        public_problem = dict(kwargs["public_problem"])
        if "target" in public_problem:
            raise ValueError("verifier target crossed into the synthetic generator")
        match = _ARITHMETIC_PATTERN.search(prompt)
        if match is None:
            raise ValueError("synthetic dry run accepts only simple arithmetic prompts")
        left, symbol, right = match.groups()
        answer = _OPERATORS[symbol](int(left), int(right))
        response = f"Check the arithmetic directly. FINAL: {answer}"
        return GenerationResult(
            response=response,
            response_tokens=tuple(response.split()),
            token_count=len(response.split()),
        )


async def run_synthetic_dry_run(
    configuration_path: str | Path,
    output_directory: str | Path,
) -> list[dict[str, Any]]:
    """Resolve config, run all local JSONL benchmarks, and persist full provenance."""
    config_path = Path(configuration_path)
    config = ExperimentConfig.from_file(config_path)
    output = Path(output_directory)
    run_store = RunArtifactStore(output)
    manifest = run_store.initialize("synthetic-dry-run", config)
    registry = default_benchmark_registry()
    checkpoint_step = config.evaluation.checkpoint_steps[0]
    model = ModelProvenance(
        model_id="synthetic-arithmetic-generator",
        checkpoint_id=f"synthetic-step-{checkpoint_step}",
        checkpoint_step=checkpoint_step,
        prompt_version="synthetic-v1",
        generation_parameters={},
    )
    summaries: list[dict[str, Any]] = []
    benchmark_versions: list[dict[str, Any]] = []
    for specification in config.evaluation.benchmarks:
        benchmark = registry.create(
            specification.adapter,
            specification.factory_configuration(base_directory=config_path.parent),
        )
        evaluator = BenchmarkEvaluator(
            JSONLEvaluationStore(output / f"{benchmark.identity.name}-records.jsonl"),
            base_seed=config.evaluation.base_seed,
            allowed_lockbox_checkpoint_steps=config.evaluation.checkpoint_steps,
        )
        report = await evaluator.evaluate(
            benchmark,
            SyntheticArithmeticGenerator(),
            model=model,
            samples_per_problem=config.evaluation.samples_per_problem,
            configuration=config.resolved_dict(),
            diagnostics_enabled=True,
            diagnostics_configuration=config.evaluation.diagnostics.model_dump(),
        )
        report_path = output / f"{benchmark.identity.name}-report.json"
        report_path.write_text(f"{report.model_dump_json()}\n", encoding="utf-8")
        summaries.append(report.aggregate.model_dump(mode="json"))
        benchmark_versions.append(benchmark.identity.model_dump(mode="json"))
    projector = None
    if config.training.algorithm is TrainingAlgorithm.SDPO:
        projector = {
            "name": "edge_local_question_feedback",
            "version": config.training.feedback.projector_version,
            "mode": config.training.feedback.mode.value,
        }
    run_store.append_checkpoint(
        manifest,
        checkpoint_step=checkpoint_step,
        model_checkpoint=model.checkpoint_id,
        teacher_identity=(
            {
                "strategy": config.training.teacher.strategy.value,
                "checkpoint_identity": config.training.teacher.checkpoint_identity,
                "version": 0,
            }
            if config.training.teacher.strategy.value != "none"
            else None
        ),
        feedback_projector=projector,
        benchmark_versions=tuple(benchmark_versions),
    )
    return summaries


def main() -> None:
    """Run the local synthetic configuration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configuration",
        nargs="?",
        default="training/configs/ood-robust-synthetic.toml",
    )
    parser.add_argument("--output", default="training/outputs/ood-robust-synthetic-dry-run")
    arguments = parser.parse_args()
    summaries = asyncio.run(run_synthetic_dry_run(arguments.configuration, arguments.output))
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
