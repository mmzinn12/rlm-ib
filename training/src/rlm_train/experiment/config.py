"""Compose immutable training, feedback, regularization, and evaluation configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlm_train.benchmarks import BenchmarkRole
from rlm_train.judge import TeacherFeedbackMode
from rlm_train.regularization import GramAnchorConfig
from rlm_train.sdpo import SDPOConfig, TeacherStrategy

EXPERIMENT_CONFIG_SCHEMA_VERSION = 1


class ImmutableConfig(BaseModel):
    """Reject unknown keys and freeze the fully resolved experiment policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingAlgorithm(StrEnum):
    """Select no training, policy-only GRPO, or SDPO-augmented policy training."""

    NONE = "none"
    GRPO = "grpo"
    SDPO = "sdpo"


class TeacherConfig(ImmutableConfig):
    """Configure fixed, EMA, or absent teacher lifecycle and exact source identity."""

    strategy: TeacherStrategy = TeacherStrategy.FIXED
    checkpoint_identity: str | None = "initial-policy"
    ema_update_rate: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_strategy(self) -> TeacherConfig:
        """Require only lifecycle-specific fields."""
        if self.strategy is TeacherStrategy.NONE:
            if self.checkpoint_identity is not None or self.ema_update_rate is not None:
                raise ValueError("no-teacher strategy cannot declare teacher fields")
        else:
            if self.checkpoint_identity is None or not self.checkpoint_identity.strip():
                raise ValueError("active teachers require checkpoint_identity")
            if self.strategy is TeacherStrategy.FIXED and self.ema_update_rate is not None:
                raise ValueError("fixed teachers do not accept ema_update_rate")
            if self.strategy is TeacherStrategy.EMA and self.ema_update_rate is None:
                raise ValueError("EMA teachers require ema_update_rate")
        return self


class FeedbackConfig(ImmutableConfig):
    """Lock one feedback projection mode and version for a complete run."""

    mode: TeacherFeedbackMode | None = TeacherFeedbackMode.DIAGNOSTIC
    projector_version: str = Field(default="v1", min_length=1)


class TrainingConfig(ImmutableConfig):
    """Compose the algorithm, teacher, projection, SDPO, and Gram policies."""

    algorithm: TrainingAlgorithm = TrainingAlgorithm.SDPO
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    sdpo: SDPOConfig | None = Field(default_factory=SDPOConfig)
    gram: GramAnchorConfig = Field(default_factory=GramAnchorConfig)

    @model_validator(mode="after")
    def validate_algorithm_components(self) -> TrainingConfig:
        """Fail fast on incompatible algorithm and auxiliary component combinations."""
        if self.algorithm is TrainingAlgorithm.SDPO:
            if self.teacher.strategy is TeacherStrategy.NONE:
                raise ValueError("SDPO requires a fixed or EMA teacher")
            if self.feedback.mode is None:
                raise ValueError("SDPO requires one teacher feedback mode")
            if self.sdpo is None:
                raise ValueError("SDPO algorithm requires sdpo component configuration")
            if self.sdpo.teacher is not self.teacher.strategy:
                raise ValueError("top-level and SDPO teacher strategies must match")
            if self.sdpo.ema_update_rate != self.teacher.ema_update_rate:
                raise ValueError("top-level and SDPO EMA update rates must match")
        else:
            if self.teacher.strategy is not TeacherStrategy.NONE:
                raise ValueError("non-SDPO algorithms cannot activate a teacher")
            if self.feedback.mode is not None:
                raise ValueError("non-SDPO algorithms cannot activate teacher feedback")
            if self.sdpo is not None:
                raise ValueError("non-SDPO algorithms cannot declare SDPO configuration")
        if self.algorithm is TrainingAlgorithm.NONE and self.gram.is_active:
            raise ValueError("base-model evaluation cannot activate a training regularizer")
        return self


class DiagnosticsConfig(ImmutableConfig):
    """Select observer-only measurements that never enter training configuration."""

    epistemic_markers: bool = True
    reasoning_dynamics: bool = True
    divergence: bool = True
    gram_drift: bool = True


class BenchmarkConfig(ImmutableConfig):
    """Describe one local generic benchmark adapter without importing its data."""

    adapter: str = "jsonl"
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    role: BenchmarkRole = BenchmarkRole.DEVELOPMENT
    path: str = Field(min_length=1)
    answer_pattern: str | None = None
    case_sensitive: bool = True

    @model_validator(mode="after")
    def validate_download_free_adapter(self) -> BenchmarkConfig:
        """Limit this implementation stage to the generic JSONL adapter."""
        if not self.adapter.strip() or not self.path.strip():
            raise ValueError("benchmark adapter and path must not be blank")
        if self.adapter != "jsonl":
            raise ValueError("only the download-free 'jsonl' adapter is currently available")
        if self.answer_pattern is not None and not self.answer_pattern.strip():
            raise ValueError("answer_pattern must not be blank")
        return self

    def factory_configuration(self, *, base_directory: str | Path | None = None) -> dict[str, Any]:
        """Return keyword arguments consumed by the generic adapter registry."""
        path = Path(self.path)
        if base_directory is not None and not path.is_absolute():
            path = Path(base_directory) / path
        return {
            "path": path,
            "name": self.name,
            "version": self.version,
            "split": self.split,
            "role": self.role,
            "answer_pattern": self.answer_pattern,
            "case_sensitive": self.case_sensitive,
        }


class EvaluationConfig(ImmutableConfig):
    """Configure deterministic checkpoint evaluation and observer diagnostics."""

    benchmarks: tuple[BenchmarkConfig, ...] = ()
    samples_per_problem: int = Field(default=1, gt=0)
    checkpoint_steps: tuple[int, ...] = Field(default=(0,), min_length=1)
    base_seed: int = Field(default=0, ge=0)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    @model_validator(mode="after")
    def validate_checkpoints(self) -> EvaluationConfig:
        """Require unique increasing checkpoint steps and benchmark identities."""
        if any(step < 0 for step in self.checkpoint_steps):
            raise ValueError("checkpoint steps must be non-negative")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint steps must be unique and increasing")
        identities = [(item.name, item.version, item.split) for item in self.benchmarks]
        if len(identities) != len(set(identities)):
            raise ValueError("evaluation benchmark identities must be unique")
        return self


class ExperimentConfig(ImmutableConfig):
    """Hold one complete, immutable, serializable experiment definition."""

    schema_version: int = EXPERIMENT_CONFIG_SCHEMA_VERSION
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_schema(self) -> ExperimentConfig:
        """Reject configurations written for another schema version."""
        if self.schema_version != EXPERIMENT_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported experiment schema version {self.schema_version}")
        return self

    def resolved_dict(self) -> dict[str, Any]:
        """Return a deep JSON-compatible mapping with every default made explicit."""
        return json.loads(self.model_dump_json())

    def canonical_json(self) -> str:
        """Serialize the complete resolved configuration deterministically."""
        return json.dumps(
            self.resolved_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        """Hash every training and evaluation choice for artifact provenance."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def training_fingerprint(self) -> str:
        """Hash only behavior-affecting training fields, excluding observers."""
        payload = json.dumps(
            self.training.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def write_resolved(self, path: str | Path) -> None:
        """Write the canonical resolved configuration to one JSON artifact."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{self.canonical_json()}\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load JSON or TOML using only the Python standard library."""
        source = Path(path)
        if source.suffix == ".toml":
            with source.open("rb") as stream:
                data = tomllib.load(stream)
        elif source.suffix == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("experiment configuration must be .toml or .json")
        return cls.model_validate(data)


__all__ = [
    "BenchmarkConfig",
    "DiagnosticsConfig",
    "EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeedbackConfig",
    "TeacherConfig",
    "TrainingAlgorithm",
    "TrainingConfig",
]
