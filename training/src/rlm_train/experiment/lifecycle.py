"""Persist resolved run and checkpoint provenance for safe resumption and comparison."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rlm_train.experiment.config import ExperimentConfig


class ImmutableArtifact(BaseModel):
    """Reject unknown artifact fields and prevent post-write mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunManifest(ImmutableArtifact):
    """Store the full resolved configuration at run initialization."""

    run_id: str = Field(min_length=1)
    preset_name: str | None = None
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_configuration: dict[str, Any]


class CheckpointProvenance(ImmutableArtifact):
    """Store every versioned component identity alongside one checkpoint."""

    run_id: str = Field(min_length=1)
    checkpoint_step: int = Field(ge=0)
    model_checkpoint: str = Field(min_length=1)
    teacher_identity: dict[str, Any] | None = None
    feedback_projector: dict[str, str] | None = None
    anchor_identity: dict[str, Any] | None = None
    benchmark_versions: tuple[dict[str, Any], ...] = ()
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_configuration: dict[str, Any]


class RunArtifactStore:
    """Write a run manifest and append unique checkpoint provenance records."""

    def __init__(self, directory: str | Path) -> None:
        if not str(directory).strip():
            raise ValueError("run artifact directory must not be blank")
        self.directory = Path(directory)
        self.manifest_path = self.directory / "run.json"
        self.checkpoints_path = self.directory / "checkpoints.jsonl"

    def initialize(
        self,
        run_id: str,
        configuration: ExperimentConfig,
        *,
        preset_name: str | None = None,
    ) -> RunManifest:
        """Create or validate the immutable run manifest for a resumable run."""
        manifest = RunManifest(
            run_id=run_id,
            preset_name=preset_name,
            configuration_fingerprint=configuration.fingerprint,
            training_fingerprint=configuration.training_fingerprint,
            resolved_configuration=configuration.resolved_dict(),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = RunManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if existing != manifest:
                raise ValueError("existing run manifest does not match requested configuration")
            return existing
        self.manifest_path.write_text(f"{manifest.model_dump_json()}\n", encoding="utf-8")
        return manifest

    def append_checkpoint(
        self,
        manifest: RunManifest,
        *,
        checkpoint_step: int,
        model_checkpoint: str,
        teacher_identity: dict[str, Any] | None = None,
        feedback_projector: dict[str, str] | None = None,
        anchor_identity: dict[str, Any] | None = None,
        benchmark_versions: tuple[dict[str, Any], ...] = (),
    ) -> CheckpointProvenance:
        """Append one checkpoint after rejecting a duplicate step."""
        artifact = CheckpointProvenance(
            run_id=manifest.run_id,
            checkpoint_step=checkpoint_step,
            model_checkpoint=model_checkpoint,
            teacher_identity=teacher_identity,
            feedback_projector=feedback_projector,
            anchor_identity=anchor_identity,
            benchmark_versions=benchmark_versions,
            configuration_fingerprint=manifest.configuration_fingerprint,
            resolved_configuration=manifest.resolved_configuration,
        )
        existing = tuple(self.iter_checkpoints())
        for item in existing:
            if item.checkpoint_step != checkpoint_step:
                continue
            if item == artifact:
                return item
            raise ValueError(f"checkpoint step {checkpoint_step} is already recorded")
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.checkpoints_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{artifact.model_dump_json()}\n")
        return artifact

    def iter_checkpoints(self) -> Iterator[CheckpointProvenance]:
        """Yield validated checkpoint records in append order."""
        if not self.checkpoints_path.exists():
            return
        with self.checkpoints_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield CheckpointProvenance.model_validate_json(line)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid checkpoint provenance on line {line_number}"
                    ) from exc


__all__ = ["CheckpointProvenance", "RunArtifactStore", "RunManifest"]
