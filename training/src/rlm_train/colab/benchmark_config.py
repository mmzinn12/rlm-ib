"""Build notebook-ready pure-SDPO configs for prepared math benchmark splits."""

from __future__ import annotations

from pathlib import Path

from rlm_train.benchmarks import BenchmarkRole, PreparedDatasetSplits
from rlm_train.colab.config import (
    ColabProfile,
    ColabRunConfig,
    DatasetConfig,
    GenerationConfig,
    JudgeConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    SDPORolloutConfig,
    TeacherRuntimeConfig,
)
from rlm_train.experiment import BenchmarkConfig, EvaluationConfig, resolve_ablation_preset

FINAL_ANSWER_PATTERN = r"FINAL:\s*(?P<answer>.+?)\s*$"


def build_benchmark_sdpo_config(
    splits: PreparedDatasetSplits,
    *,
    run_name: str,
    max_optimizer_steps: int = 100,
    checkpoint_every_steps: int = 20,
    evaluate_every_steps: int = 20,
    google_drive_root: str | None = "/content/drive/MyDrive",
    output_directory: str = "rlm-ib-runs",
    seed: int = 17,
    model: ModelConfig | None = None,
    generation: GenerationConfig | None = None,
    sdpo_rollout: SDPORolloutConfig | None = None,
    judge: JudgeConfig | None = None,
    teacher_runtime: TeacherRuntimeConfig | None = None,
) -> ColabRunConfig:
    """Create an isolated train/eval run with SDPO loss and no GRPO policy term."""
    if not run_name.strip():
        raise ValueError("benchmark SDPO run_name must not be blank")
    evaluation = EvaluationConfig(
        benchmarks=(
            BenchmarkConfig(
                adapter="jsonl",
                name=f"{splits.name}-test",
                version=splits.source.revision,
                split="test",
                role=BenchmarkRole.DEVELOPMENT,
                path=str(splits.test_path),
                answer_pattern=FINAL_ANSWER_PATTERN,
                case_sensitive=True,
            ),
        ),
        samples_per_problem=1,
        checkpoint_steps=(0,),
        base_seed=seed,
    )
    return ColabRunConfig(
        profile=ColabProfile.TRAIN,
        experiment_preset=None,
        experiment=resolve_ablation_preset("edge_local_sdpo", evaluation=evaluation),
        model=model or ModelConfig(),
        generation=generation
        or GenerationConfig(
            system_prompt=(
                "Solve the problem carefully and end with FINAL: followed by the answer."
            ),
            max_prompt_tokens=512,
            max_new_tokens=256,
            rollouts_per_prompt=1,
        ),
        sdpo_rollout=sdpo_rollout or SDPORolloutConfig(),
        optimization=OptimizationConfig(
            learning_rate=2e-4,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_optimizer_steps=max_optimizer_steps,
            warmup_steps=min(5, max_optimizer_steps),
            scheduler="cosine",
            policy_weight=0.0,
            sdpo_weight=1.0,
            gram_weight=0.0,
            kl_coefficient=0.0,
        ),
        dataset=DatasetConfig(
            path=str(splits.train_path),
            name=f"{splits.name}-train",
            version=splits.source.revision,
            split="train",
            answer_pattern=FINAL_ANSWER_PATTERN,
            case_sensitive=True,
            rubric="exact_match",
        ),
        judge=judge or JudgeConfig(),
        teacher_runtime=teacher_runtime or TeacherRuntimeConfig(),
        output=OutputConfig(
            output_directory=output_directory,
            google_drive_root=google_drive_root,
            run_name=run_name,
            checkpoint_every_steps=checkpoint_every_steps,
            evaluate_every_steps=evaluate_every_steps,
        ),
        seed=seed,
    )


def write_colab_run_config(configuration: ColabRunConfig, path: str | Path) -> Path:
    """Atomically write a validated notebook config and return its absolute path."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(f"{configuration.model_dump_json(indent=2)}\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = [
    "FINAL_ANSWER_PATTERN",
    "build_benchmark_sdpo_config",
    "write_colab_run_config",
]
