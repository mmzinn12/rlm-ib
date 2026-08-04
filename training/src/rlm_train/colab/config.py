"""Immutable configuration for the standalone single-GPU Transformers trainer."""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlm_train.experiment import ABLATION_PRESETS, ExperimentConfig, resolve_ablation_preset
from rlm_train.experiment.config import TrainingAlgorithm


class ImmutableConfig(BaseModel):
    """Reject unknown fields and freeze resolved runtime policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ColabProfile(StrEnum):
    """Select a one-step environment check or a real training run."""

    SMOKE = "smoke"
    TRAIN = "train"


class Precision(StrEnum):
    """Numerical precision used by model loading and autocast."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class Quantization(StrEnum):
    """Optional bitsandbytes base-model quantization."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


class TeacherResidency(StrEnum):
    """Single-GPU placement strategy for an immutable teacher."""

    RESIDENT = "resident"
    CPU_OFFLOAD = "cpu_offload"
    SEQUENTIAL = "sequential"


class ModelConfig(ImmutableConfig):
    """Select exact model/tokenizer sources and LoRA adapter parameters."""

    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_revision: str = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    precision: Precision = Precision.FP16
    quantization: Quantization = Quantization.NONE
    max_context_length: int = Field(default=2048, gt=0)
    lora_rank: int = Field(default=8, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    @model_validator(mode="after")
    def validate_model(self) -> ModelConfig:
        """Reject blank revisions and invalid adapter selections."""
        values = [self.model_id, self.model_revision]
        if self.tokenizer_id is not None:
            values.append(self.tokenizer_id)
        if self.tokenizer_revision is not None:
            values.append(self.tokenizer_revision)
        if any(not value.strip() for value in values):
            raise ValueError("model and tokenizer identities/revisions must not be blank")
        if not self.lora_target_modules or any(
            not name.strip() for name in self.lora_target_modules
        ):
            raise ValueError("LoRA target modules must not be empty or blank")
        return self

    @property
    def resolved_tokenizer_id(self) -> str:
        """Use the model tokenizer unless an explicit compatible tokenizer is supplied."""
        return self.tokenizer_id or self.model_id

    @property
    def resolved_tokenizer_revision(self) -> str:
        """Use the model revision unless a tokenizer revision is supplied."""
        return self.tokenizer_revision or self.model_revision


class GenerationConfig(ImmutableConfig):
    """Configure exact prompt formatting and grouped response sampling."""

    prompt_template_version: str = "chat-v1"
    system_prompt: str = "Solve the problem carefully and give a final answer."
    use_chat_template: bool = True
    max_prompt_tokens: int = Field(default=512, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)
    rollouts_per_prompt: int = Field(default=4, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    do_sample: bool = True
    allow_prompt_truncation: bool = False

    @model_validator(mode="after")
    def validate_prompt_template(self) -> GenerationConfig:
        """Require versioned, non-blank prompt policy."""
        if not self.prompt_template_version.strip() or not self.system_prompt.strip():
            raise ValueError("prompt-template version and system prompt must not be blank")
        return self


class SDPORolloutConfig(ImmutableConfig):
    """Configure the explicit helper-question edge sampled for SDPO training."""

    question_system_prompt: str = (
        "Given the problem, respond with exactly one useful helper question and nothing else."
    )
    child_system_prompt: str = (
        "Answer the helper question carefully using only the problem and question provided."
    )
    max_question_tokens: int = Field(default=64, gt=0)
    max_child_tokens: int = Field(default=128, gt=0)

    @model_validator(mode="after")
    def validate_prompts(self) -> SDPORolloutConfig:
        """Require explicit non-blank policies for both nodes in the traced edge."""
        if not self.question_system_prompt.strip() or not self.child_system_prompt.strip():
            raise ValueError("SDPO question and child system prompts must not be blank")
        return self


class OptimizationConfig(ImmutableConfig):
    """Configure the local policy optimizer and explicit objective coefficients."""

    learning_rate: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    epsilon: float = Field(default=1e-8, gt=0.0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_optimizer_steps: int = Field(default=100, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    scheduler: Literal["constant", "linear", "cosine"] = "linear"
    max_gradient_norm: float = Field(default=1.0, gt=0.0)
    grpo_clip_epsilon: float = Field(default=0.2, gt=0.0)
    policy_weight: float = Field(default=1.0, ge=0.0)
    sdpo_weight: float = Field(default=0.0, ge=0.0)
    gram_weight: float = Field(default=0.0, ge=0.0)
    kl_coefficient: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_objective(self) -> OptimizationConfig:
        """Require a real objective and valid warmup horizon."""
        if self.policy_weight + self.sdpo_weight + self.gram_weight <= 0.0:
            raise ValueError("at least one training objective coefficient must be positive")
        if self.warmup_steps > self.max_optimizer_steps:
            raise ValueError("warmup_steps cannot exceed max_optimizer_steps")
        return self


class DatasetConfig(ImmutableConfig):
    """Select a local generic JSONL training dataset."""

    path: str = "training/benchmarks/synthetic-arithmetic.jsonl"
    name: str = "synthetic-arithmetic-train"
    version: str = "v1"
    split: str = "train"
    answer_pattern: str | None = r"FINAL:\s*(?P<answer>-?\d+)\s*$"
    case_sensitive: bool = True
    rubric: Literal["exact_match", "numeric_proximity", "smoke_index"] = "exact_match"

    @model_validator(mode="after")
    def validate_dataset(self) -> DatasetConfig:
        """Require stable non-empty dataset identity fields."""
        if any(not value.strip() for value in (self.path, self.name, self.version, self.split)):
            raise ValueError("dataset path and identity fields must not be blank")
        return self


class JudgeConfig(ImmutableConfig):
    """Select a deterministic fake judge or an API-backed structured judge."""

    provider: Literal["fake", "openai"] = "fake"
    model: str = "deterministic-fake"
    model_revision: str = "v1"
    api_key_environment: str = "OPENAI_API_KEY"
    max_attempts: int = Field(default=2, gt=0, le=5)
    prompt_schema_version: str = "trajectory-feedback-v1"

    @model_validator(mode="after")
    def validate_judge(self) -> JudgeConfig:
        """Reject blank API and schema identifiers without storing credentials."""
        values = (
            self.model,
            self.model_revision,
            self.api_key_environment,
            self.prompt_schema_version,
        )
        if any(not value.strip() for value in values):
            raise ValueError("judge identity, secret name, and schema version must not be blank")
        return self


class TeacherRuntimeConfig(ImmutableConfig):
    """Configure teacher placement and immutable target caching."""

    residency: TeacherResidency = TeacherResidency.CPU_OFFLOAD
    cache_directory: str = "teacher-targets"
    fingerprint_interval: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_teacher_runtime(self) -> TeacherRuntimeConfig:
        """Require a non-blank cache location."""
        if not self.cache_directory.strip():
            raise ValueError("teacher cache directory must not be blank")
        return self


class OutputConfig(ImmutableConfig):
    """Select local or optional Google Drive run storage."""

    output_directory: str = "training/outputs/colab"
    google_drive_root: str | None = None
    run_name: str = "single-gpu"
    checkpoint_every_steps: int = Field(default=20, gt=0)
    evaluate_every_steps: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_output(self) -> OutputConfig:
        """Reject blank paths and run identifiers."""
        values = [self.output_directory, self.run_name]
        if self.google_drive_root is not None:
            values.append(self.google_drive_root)
        if any(not value.strip() for value in values):
            raise ValueError("output paths and run name must not be blank")
        if self.google_drive_root is not None and Path(self.output_directory).is_absolute():
            raise ValueError("Google Drive output_directory must be relative to its root")
        return self

    def resolve_directory(self, *, base_directory: str | Path | None = None) -> Path:
        """Resolve an explicit run directory without requiring Google Drive."""
        relative = Path(self.output_directory)
        if self.google_drive_root is not None:
            root = Path(self.google_drive_root).expanduser()
        elif base_directory is not None and not relative.is_absolute():
            root = Path(base_directory)
        else:
            root = Path()
        return (root / relative / self.run_name).expanduser().resolve()


class ColabRunConfig(ImmutableConfig):
    """Compose an experiment schema with its standalone Transformers runtime."""

    schema_version: int = 1
    profile: ColabProfile = ColabProfile.SMOKE
    execution_backend: Literal["transformers"] = "transformers"
    experiment_preset: Literal[*ABLATION_PRESETS] | None = "edge_local_sdpo"
    experiment: ExperimentConfig | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(
        default_factory=lambda: GenerationConfig(
            max_prompt_tokens=256,
            max_new_tokens=32,
            rollouts_per_prompt=2,
        )
    )
    sdpo_rollout: SDPORolloutConfig = Field(default_factory=SDPORolloutConfig)
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(max_optimizer_steps=1, sdpo_weight=1.0)
    )
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig(rubric="smoke_index"))
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    teacher_runtime: TeacherRuntimeConfig = Field(default_factory=TeacherRuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_run(self) -> ColabRunConfig:
        """Align runtime objectives with the selected high-level experiment arm."""
        if self.schema_version != 1:
            raise ValueError(f"unsupported Colab run schema version {self.schema_version}")
        if (self.experiment is None) == (self.experiment_preset is None):
            raise ValueError("select exactly one of experiment or experiment_preset")
        experiment = self.resolved_experiment
        algorithm = experiment.training.algorithm
        if algorithm is TrainingAlgorithm.NONE:
            raise ValueError("the Colab trainer requires grpo or sdpo, not base evaluation")
        if algorithm is TrainingAlgorithm.GRPO:
            if self.optimization.policy_weight <= 0.0:
                raise ValueError("GRPO runs require a positive policy_weight")
            if self.optimization.sdpo_weight != 0.0:
                raise ValueError("GRPO cannot activate the SDPO loss")
        if algorithm is TrainingAlgorithm.SDPO and self.optimization.sdpo_weight <= 0.0:
            raise ValueError("SDPO runs require a positive sdpo_weight")
        if self.optimization.gram_weight > 0.0 and not experiment.training.gram.is_active:
            raise ValueError("gram_weight requires an active top-level Gram configuration")
        if self.optimization.gram_weight == 0.0 and experiment.training.gram.is_active:
            raise ValueError("an active top-level Gram configuration requires gram_weight")
        if (
            experiment.training.gram.is_active
            and self.optimization.gram_weight != experiment.training.gram.loss_weight
        ):
            raise ValueError("runtime and top-level Gram loss weights must match")
        if self.profile is ColabProfile.SMOKE:
            if self.optimization.max_optimizer_steps != 1:
                raise ValueError("the smoke profile must run exactly one optimizer step")
            if self.generation.rollouts_per_prompt > 2 or self.generation.max_new_tokens > 64:
                raise ValueError("the smoke profile requires at most two short rollouts")
            if self.dataset.rubric != "smoke_index":
                raise ValueError("the smoke profile requires the explicit smoke_index rubric")
        elif self.dataset.rubric == "smoke_index":
            raise ValueError("smoke_index is an environment test rubric, not a train rubric")
        if self.generation.max_prompt_tokens + self.generation.max_new_tokens > (
            self.model.max_context_length
        ):
            raise ValueError("prompt and continuation limits exceed model context length")
        if (
            algorithm is TrainingAlgorithm.SDPO
            and self.generation.max_prompt_tokens
            + max(
                self.sdpo_rollout.max_question_tokens,
                self.sdpo_rollout.max_child_tokens,
            )
            > self.model.max_context_length
        ):
            raise ValueError("SDPO prompt and node continuation limits exceed model context length")
        return self

    @property
    def resolved_experiment(self) -> ExperimentConfig:
        """Resolve named presets through the same explicit component schema."""
        if self.experiment is not None:
            return self.experiment
        assert self.experiment_preset is not None
        return resolve_ablation_preset(self.experiment_preset)

    def resolved_dict(self, *, base_directory: str | Path | None = None) -> dict[str, Any]:
        """Return complete, secret-free configuration and an explicit output path."""
        payload = self.model_dump(mode="json", exclude={"experiment"})
        payload["experiment"] = self.resolved_experiment.resolved_dict()
        payload["resolved_output_directory"] = str(
            self.output.resolve_directory(base_directory=base_directory)
        )
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))

    def canonical_json(self, *, base_directory: str | Path | None = None) -> str:
        """Serialize the complete resolved Colab run deterministically."""
        return json.dumps(
            self.resolved_dict(base_directory=base_directory),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def fingerprint(self, *, base_directory: str | Path | None = None) -> str:
        """Fingerprint behavior and the resolved storage target."""
        return hashlib.sha256(
            self.canonical_json(base_directory=base_directory).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> ColabRunConfig:
        """Load a JSON or TOML single-GPU configuration."""
        source = Path(path)
        if source.suffix == ".toml":
            with source.open("rb") as stream:
                data = tomllib.load(stream)
        elif source.suffix == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("Colab configuration must be .toml or .json")
        return cls.model_validate(data)


__all__ = [
    "ColabProfile",
    "ColabRunConfig",
    "DatasetConfig",
    "GenerationConfig",
    "JudgeConfig",
    "ModelConfig",
    "OptimizationConfig",
    "OutputConfig",
    "Precision",
    "Quantization",
    "SDPORolloutConfig",
    "TeacherResidency",
    "TeacherRuntimeConfig",
]
