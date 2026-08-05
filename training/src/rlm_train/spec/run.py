"""Immutable, serializable top-level training and evaluation RunSpec."""

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
    adapter: str = "jsonl"
    source: str = Field(min_length=1)
    split: str = Field(default="train", min_length=1)
    name: str | None = None
    version: str | None = None


class RuntimeSpec(ImmutableSpec):
    device: str = "auto"
    precision: str = "fp32"
    seed: int = Field(default=0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_optimizer_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_gradient_norm: float = Field(default=1.0, gt=0.0)


class RunSpec(ImmutableSpec):
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
