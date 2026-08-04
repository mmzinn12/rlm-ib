"""Persist versioned trajectory, feedback, and reproducibility artifacts as JSONL.

Purpose:
    Make completed rollouts reusable for judge iteration, compilation, tokenization,
    and loss debugging without invoking the student again.
Implementation:
    A strict dataclass owns the versioned wire payload. ``JSONLTrajectoryStore`` appends
    unique artifacts and validates every record while reading. Privileged judge content
    is never stored; only its source/version/fingerprint descriptor is permitted.
Inputs:
    Task and dataset identity, a complete trajectory, optional judge feedback, model
    and tokenizer identity, configurations, seeds, and lifecycle metadata.
Outputs:
    Validated ``TrajectoryArtifact`` values and deterministic JSONL records.
Example:
    ``JSONLTrajectoryStore("rollouts.jsonl").append(artifact)``
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from rlm.core.trajectory import TrajectoryTree

from rlm_train.judge.base import TaskContext
from rlm_train.judge.context import PrivilegedContextDescriptor
from rlm_train.judge.schema import TrajectoryFeedback

TRAJECTORY_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryArtifact:
    """Store one completed rollout and all code-level replay inputs.

    Required identities are explicit. Optional configuration and lifecycle maps remain
    framework-neutral until a version-pinned trainer adapter consumes them.
    """

    artifact_id: str
    task_id: str
    task_prompt: Any
    student_model: str
    tokenizer_fingerprint: str
    trajectory: TrajectoryTree
    schema_version: int = TRAJECTORY_ARTIFACT_SCHEMA_VERSION
    dataset_id: str | None = None
    dataset_revision: str | None = None
    example_id: str | None = None
    context_references: tuple[str, ...] = ()
    task_evidence_snapshot: Any = None
    task_metadata: dict[str, Any] = field(default_factory=dict)
    student_checkpoint: str | None = None
    policy_version: int | None = None
    trainer_configuration: dict[str, Any] = field(default_factory=dict)
    inference_configuration: dict[str, Any] = field(default_factory=dict)
    experiment_configuration: dict[str, Any] | None = None
    feedback_projector: dict[str, str] | None = None
    feedback: TrajectoryFeedback | None = None
    privileged_context: PrivilegedContextDescriptor | None = None
    anchor_identity: dict[str, Any] | None = None
    teacher_identity: dict[str, Any] | None = None
    sampling_seeds: dict[str, int] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identities, versions, trajectory references, and JSON safety."""
        if self.schema_version != TRAJECTORY_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported trajectory artifact schema version {self.schema_version}"
            )
        if not self.artifact_id.strip() or not self.task_id.strip():
            raise ValueError("artifact_id and task_id must not be blank")
        if self.dataset_id is not None and not self.dataset_id.strip():
            raise ValueError("dataset_id must not be blank when provided")
        if self.dataset_revision is not None and not self.dataset_revision.strip():
            raise ValueError("dataset_revision must not be blank when provided")
        if self.example_id is not None and not self.example_id.strip():
            raise ValueError("example_id must not be blank when provided")
        if not self.student_model.strip() or not self.tokenizer_fingerprint.strip():
            raise ValueError("student model and tokenizer fingerprint must not be blank")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if any(not name.strip() or seed < 0 for name, seed in self.sampling_seeds.items()):
            raise ValueError("sampling seeds require non-blank names and non-negative values")
        if any(not reference.strip() for reference in self.context_references):
            raise ValueError("context references must not contain blank values")
        if self.feedback_projector is not None:
            if set(self.feedback_projector) != {"name", "version", "mode"}:
                raise ValueError("feedback_projector requires name, version, and mode")
            if any(not value.strip() for value in self.feedback_projector.values()):
                raise ValueError("feedback_projector values must not be blank")
        self.trajectory.validate()
        if self.feedback is not None:
            self.feedback.validate_node_ids({node.node_id for node in self.trajectory.nodes})
        ensure_json_compatible(self.wire_payload(), name="trajectory artifact")

    @classmethod
    def from_task(
        cls,
        *,
        artifact_id: str,
        task: TaskContext,
        student_model: str,
        tokenizer_fingerprint: str,
        trajectory: TrajectoryTree,
        **kwargs: Any,
    ) -> TrajectoryArtifact:
        """Create an artifact from a judge task without storing privileged content."""
        public = task.public_payload()
        return cls(
            artifact_id=artifact_id,
            task_id=task.task_id,
            task_prompt=public["prompt"],
            task_evidence_snapshot=public["evidence_snapshot"],
            task_metadata=public["metadata"],
            student_model=student_model,
            tokenizer_fingerprint=tokenizer_fingerprint,
            trajectory=trajectory,
            privileged_context=task.privileged_descriptor(),
            **kwargs,
        )

    def with_feedback(self, feedback: TrajectoryFeedback) -> TrajectoryArtifact:
        """Return a revalidated copy containing replaced judge feedback."""
        return replace(self, feedback=feedback)

    def wire_payload(self) -> dict[str, Any]:
        """Build the unnormalized versioned wire mapping."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "example_id": self.example_id,
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "context_references": list(self.context_references),
            "task_evidence_snapshot": self.task_evidence_snapshot,
            "task_metadata": self.task_metadata,
            "student_model": self.student_model,
            "student_checkpoint": self.student_checkpoint,
            "policy_version": self.policy_version,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "trainer_configuration": self.trainer_configuration,
            "inference_configuration": self.inference_configuration,
            "experiment_configuration": self.experiment_configuration,
            "feedback_projector": self.feedback_projector,
            "trajectory": self.trajectory.to_dict(),
            "feedback": self.feedback.model_dump(mode="json") if self.feedback else None,
            "privileged_context": (
                self.privileged_context.to_dict() if self.privileged_context else None
            ),
            "anchor_identity": self.anchor_identity,
            "teacher_identity": self.teacher_identity,
            "sampling_seeds": self.sampling_seeds,
            "run_metadata": self.run_metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, normalized JSON-compatible artifact mapping."""
        return json.loads(ensure_json_compatible(self.wire_payload(), name="trajectory artifact"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryArtifact:
        """Reconstruct an artifact and reject unknown wire fields."""
        allowed = {
            "schema_version",
            "artifact_id",
            "dataset_id",
            "dataset_revision",
            "example_id",
            "task_id",
            "task_prompt",
            "context_references",
            "task_evidence_snapshot",
            "task_metadata",
            "student_model",
            "student_checkpoint",
            "policy_version",
            "tokenizer_fingerprint",
            "trainer_configuration",
            "inference_configuration",
            "experiment_configuration",
            "feedback_projector",
            "trajectory",
            "feedback",
            "privileged_context",
            "anchor_identity",
            "teacher_identity",
            "sampling_seeds",
            "run_metadata",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown trajectory artifact fields: {sorted(unknown)!r}")
        feedback_data = data.get("feedback")
        privileged_data = data.get("privileged_context")
        return cls(
            schema_version=int(data["schema_version"]),
            artifact_id=str(data["artifact_id"]),
            dataset_id=data.get("dataset_id"),
            dataset_revision=data.get("dataset_revision"),
            example_id=data.get("example_id"),
            task_id=str(data["task_id"]),
            task_prompt=data.get("task_prompt"),
            context_references=tuple(data.get("context_references") or ()),
            task_evidence_snapshot=data.get("task_evidence_snapshot"),
            task_metadata=dict(data.get("task_metadata") or {}),
            student_model=str(data["student_model"]),
            student_checkpoint=data.get("student_checkpoint"),
            policy_version=data.get("policy_version"),
            tokenizer_fingerprint=str(data["tokenizer_fingerprint"]),
            trainer_configuration=dict(data.get("trainer_configuration") or {}),
            inference_configuration=dict(data.get("inference_configuration") or {}),
            experiment_configuration=(
                dict(data["experiment_configuration"])
                if data.get("experiment_configuration") is not None
                else None
            ),
            feedback_projector=(
                {str(key): str(value) for key, value in data["feedback_projector"].items()}
                if data.get("feedback_projector") is not None
                else None
            ),
            trajectory=TrajectoryTree.from_dict(dict(data["trajectory"])),
            feedback=(
                TrajectoryFeedback.model_validate(feedback_data)
                if feedback_data is not None
                else None
            ),
            privileged_context=(
                PrivilegedContextDescriptor.from_dict(dict(privileged_data))
                if privileged_data is not None
                else None
            ),
            anchor_identity=(
                dict(data["anchor_identity"]) if data.get("anchor_identity") is not None else None
            ),
            teacher_identity=(
                dict(data["teacher_identity"]) if data.get("teacher_identity") is not None else None
            ),
            sampling_seeds={
                str(name): int(seed) for name, seed in (data.get("sampling_seeds") or {}).items()
            },
            run_metadata=dict(data.get("run_metadata") or {}),
        )


def ensure_json_compatible(value: Any, *, name: str) -> str:
    """Serialize finite JSON data or raise a public validation error."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON-compatible values") from exc


class TrajectoryArtifactStore(Protocol):
    """Define append and iteration operations for rollout artifact storage."""

    def append(self, artifact: TrajectoryArtifact) -> None:
        """Persist one artifact with a unique ID."""
        ...

    def iter_artifacts(self) -> Iterator[TrajectoryArtifact]:
        """Yield validated artifacts in storage order."""
        ...


class JSONLTrajectoryStore:
    """Persist unique, versioned trajectory artifacts in an append-only JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Configure a JSONL path without creating an empty file eagerly."""
        if not str(path).strip():
            raise ValueError("trajectory store path must not be blank")
        self.path = Path(path)

    def append(self, artifact: TrajectoryArtifact) -> None:
        """Append one artifact after rejecting duplicate artifact IDs."""
        if any(existing.artifact_id == artifact.artifact_id for existing in self.iter_artifacts()):
            raise ValueError(f"duplicate trajectory artifact ID {artifact.artifact_id!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = ensure_json_compatible(artifact.to_dict(), name="trajectory artifact")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{line}\n")

    def iter_artifacts(self) -> Iterator[TrajectoryArtifact]:
        """Yield validated records and report corrupt line numbers loudly."""
        if not self.path.exists():
            return
        seen_ids: set[str] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    artifact = TrajectoryArtifact.from_dict(payload)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid trajectory artifact at {self.path}:{line_number}"
                    ) from exc
                if artifact.artifact_id in seen_ids:
                    raise ValueError(
                        f"duplicate trajectory artifact ID {artifact.artifact_id!r} in store"
                    )
                seen_ids.add(artifact.artifact_id)
                yield artifact

    def get(self, artifact_id: str) -> TrajectoryArtifact | None:
        """Return one artifact by ID, or ``None`` when absent."""
        if not artifact_id:
            raise ValueError("artifact_id must not be empty")
        return next(
            (artifact for artifact in self.iter_artifacts() if artifact.artifact_id == artifact_id),
            None,
        )


__all__ = [
    "JSONLTrajectoryStore",
    "TRAJECTORY_ARTIFACT_SCHEMA_VERSION",
    "TrajectoryArtifact",
    "TrajectoryArtifactStore",
]
