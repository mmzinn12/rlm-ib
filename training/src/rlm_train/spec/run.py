"""Immutable, serializable top-level training and evaluation RunSpec.

This module defines the canonical configuration tree for a run. ``RunSpec`` is the single
source of truth consumed by every ``build_*`` entry point and by the CLI; it is frozen,
JSON/TOML round-trippable, and content-addressable via ``fingerprint``. ``DatasetRefSpec``
and ``RuntimeSpec`` are the two leaf specs defined here; the remaining sub-specs live in
sibling modules under ``rlm_train.spec``.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from rlm_train.spec.artifacts import ArtifactSpec
from rlm_train.spec.evaluation import EvaluationSpec
from rlm_train.spec.feedback import AssessmentScope, FeedbackSpec
from rlm_train.spec.models import ImmutableSpec, JudgeSpec, StudentSpec, TeacherSpec
from rlm_train.spec.objectives import ObjectivesSpec
from rlm_train.spec.rollout import RolloutSpec

RUN_SPEC_SCHEMA_VERSION = 1


class DatasetRefSpec(ImmutableSpec):
    """Reference to a dataset the run should load, independent of any adapter internals.

    Attributes:
        adapter: Dataset loader name: ``"jsonl"`` or ``"hotpotqa"``.
        source: Path or URI to the dataset the adapter reads from.
        split: Named split to load, e.g. ``"train"`` or ``"test"``.
        subset: Optional Hugging Face dataset configuration/subset name.
        revision: Optional immutable Hugging Face dataset revision.
        max_records: Optional deterministic prefix size, useful for bounded Colab runs.
        name: Optional human-facing dataset identifier for provenance.
        version: Optional dataset version tag recorded for reproducibility.
    """

    adapter: str = "jsonl"
    source: str = Field(min_length=1)
    split: str = Field(default="train", min_length=1)
    subset: str | None = None
    revision: str | None = None
    max_records: int | None = Field(default=None, gt=0)
    name: str | None = None
    version: str | None = None


class RuntimeSpec(ImmutableSpec):
    """Device, precision, and optimization settings that drive the training loop.

    Attributes:
        device: Target device selector, e.g. ``"auto"``, ``"cpu"``, or ``"cuda"``.
        precision: Model numeric precision: ``"fp32"``, ``"bf16"``, or ``"fp16"``.
        seed: Base RNG seed for deterministic sampling and initialization.
        gradient_accumulation_steps: Micro-batches accumulated before each optimizer step.
        max_optimizer_steps: Total number of optimizer steps to run.
        learning_rate: Peak learning rate passed to the optimizer.
        max_gradient_norm: Gradient-norm clipping threshold applied before each step.
        warmup_steps: Linear-warmup steps for the scheduler; ``0`` disables it (constant LR).
        scheduler: Learning-rate schedule shape: ``"constant"``, ``"linear"``, or ``"cosine"``.
    """

    device: str = "auto"
    precision: str = "fp32"
    seed: int = Field(default=0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_optimizer_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_gradient_norm: float = Field(default=1.0, gt=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    scheduler: str = "linear"

    @model_validator(mode="after")
    def validate_schedule(self) -> RuntimeSpec:
        if self.scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("scheduler must be constant, linear, or cosine")
        if self.warmup_steps > self.max_optimizer_steps:
            raise ValueError("warmup_steps cannot exceed max_optimizer_steps")
        return self


class RunSpec(ImmutableSpec):
    """Immutable top-level configuration for a full training and evaluation run.

    ``RunSpec`` is the single source of truth every ``build_*`` entry point reads. It is frozen,
    validated on construction, and serialized verbatim to JSON/TOML for the CLI.

    Attributes:
        schema_version: Version of the RunSpec schema; validated for forward compatibility.
        student: Trainable student policy (model, tokenizer, adapter) specification.
        rollout: RLM rollout engine configuration (environment, depth, sampling).
        judge: Structured judge configuration used to produce feedback.
        teacher: Teacher strategy that supplies distillation targets.
        feedback: Feedback visibility policy and privileged-hindsight opt-in.
        objectives: Enabled training objectives (SDPO/GRPO/Gram) and their weights.
        training_dataset: Dataset used for training; ``None`` for evaluation-only runs.
        evaluation_datasets: Held-out datasets scored during evaluation.
        evaluation: Whole-recursive-policy evaluation settings.
        artifacts: Output directory and artifact retention configuration.
        runtime: Device, precision, and optimization settings (see ``RuntimeSpec``).
    """

    schema_version: int = RUN_SPEC_SCHEMA_VERSION
    student: StudentSpec
    rollout: RolloutSpec = Field(default_factory=RolloutSpec)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    teacher: TeacherSpec = Field(default_factory=TeacherSpec)
    feedback: FeedbackSpec = Field(default_factory=FeedbackSpec)
    objectives: ObjectivesSpec = Field(default_factory=ObjectivesSpec)
    training_dataset: DatasetRefSpec | None = None
    evaluation_datasets: tuple[DatasetRefSpec, ...] = ()
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    artifacts: ArtifactSpec = Field(default_factory=ArtifactSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    @model_validator(mode="after")
    def validate_run(self) -> RunSpec:
        if self.schema_version != RUN_SPEC_SCHEMA_VERSION:
            raise ValueError(f"unsupported RunSpec schema version {self.schema_version}")
        if self.objectives.enabled and not self.student.trainable:
            raise ValueError("enabled objectives require a trainable student")
        for name, objective in self.objectives.enabled:
            scope = getattr(objective, "feedback_scope", None)
            if (
                scope is AssessmentScope.PRIVILEGED_DIAGNOSTIC
                and not self.feedback.allow_privileged_hindsight_distillation
            ):
                raise ValueError(
                    f"{name} requests privileged feedback without the explicit hindsight opt-in"
                )
        return self

    def resolved_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json(by_alias=True))

    def canonical_json(self) -> str:
        return json.dumps(
            self.resolved_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_file(cls, path: str | Path) -> RunSpec:
        source = Path(path)
        if source.suffix == ".toml":
            with source.open("rb") as stream:
                payload = tomllib.load(stream)
        elif source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("RunSpec must be loaded from .toml or .json")
        return cls.model_validate(payload)

    def write_resolved(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"{self.canonical_json()}\n", encoding="utf-8")


__all__ = ["DatasetRefSpec", "RUN_SPEC_SCHEMA_VERSION", "RunSpec", "RuntimeSpec"]
