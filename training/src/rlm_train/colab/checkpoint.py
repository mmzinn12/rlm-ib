"""Atomic, fingerprint-validated checkpoints for local LoRA training and evaluation."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rlm_train.colab.config import ColabRunConfig
from rlm_train.colab.trainer import SingleGPUTrainer, TrainerState


class CheckpointManifest(BaseModel):
    """Identify immutable run inputs and every resumable payload in one checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int = 1
    global_step: int = Field(ge=0)
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_configuration: dict[str, Any]
    model_identity: dict[str, Any]
    tokenizer_fingerprint: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_fingerprints: dict[str, str] = Field(default_factory=dict)
    overlap_check_results: dict[str, Any] = Field(default_factory=dict)
    teacher_identity: dict[str, Any] | None = None
    anchor_identity: dict[str, Any] | None = None
    judge_identity: dict[str, Any] | None = None
    judge_cache_manifest: dict[str, Any] | None = None
    teacher_cache_manifest: dict[str, Any] | None = None
    completed_evaluation_keys: tuple[str, ...] = ()
    source_revision: str = "unknown"


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Return validated checkpoint metadata and optional controller state."""

    path: Path
    manifest: CheckpointManifest
    controller_state: Any | None


class TrainingCheckpointManager:
    """Save independent checkpoints and maintain explicit latest/best pointers."""

    def __init__(
        self,
        run_directory: str | Path,
        configuration: ColabRunConfig,
        *,
        base_directory: str | Path | None = None,
    ) -> None:
        if not str(run_directory).strip():
            raise ValueError("checkpoint run directory must not be blank")
        self.run_directory = Path(run_directory)
        self.configuration = configuration
        self.base_directory = base_directory
        self.resolved_configuration = configuration.resolved_dict(base_directory=base_directory)
        self.configuration_fingerprint = configuration.fingerprint(base_directory=base_directory)

    def save(
        self,
        trainer: SingleGPUTrainer,
        *,
        model_identity: Mapping[str, Any],
        tokenizer_fingerprint: str,
        dataset_fingerprint: str,
        benchmark_fingerprints: Mapping[str, str] | None = None,
        overlap_check_results: Mapping[str, Any] | None = None,
        teacher_identity: Mapping[str, Any] | None = None,
        anchor_identity: Mapping[str, Any] | None = None,
        judge_identity: Mapping[str, Any] | None = None,
        judge_cache_manifest: Mapping[str, Any] | None = None,
        teacher_cache_manifest: Mapping[str, Any] | None = None,
        completed_evaluation_keys: tuple[str, ...] = (),
        controller_state: Any | None = None,
        source_revision: str = "unknown",
        is_best: bool = False,
    ) -> Path:
        """Atomically persist trainable weights, optimizer state, RNG, and provenance."""
        torch = _torch()
        step = trainer.state.global_step
        destination = self.run_directory / f"checkpoint-{step:08d}"
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        self.run_directory.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".checkpoint-{step:08d}-", dir=self.run_directory)
        )
        try:
            trainable_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in trainer.trainable_parameters.items()
            }
            torch.save(trainable_state, temporary / "student_adapter.pt")
            torch.save(
                {
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": trainer.scheduler.state_dict(),
                    "scaler": trainer.scaler.state_dict(),
                    "trainer": trainer.state.to_dict(),
                    "rng": capture_rng_state(),
                    "controller": controller_state,
                },
                temporary / "training_state.pt",
            )
            manifest = CheckpointManifest(
                global_step=step,
                configuration_fingerprint=self.configuration_fingerprint,
                resolved_configuration=self.resolved_configuration,
                model_identity=json.loads(json.dumps(model_identity, allow_nan=False)),
                tokenizer_fingerprint=tokenizer_fingerprint,
                dataset_fingerprint=dataset_fingerprint,
                benchmark_fingerprints=dict(benchmark_fingerprints or {}),
                overlap_check_results=json.loads(
                    json.dumps(overlap_check_results or {}, allow_nan=False)
                ),
                teacher_identity=(dict(teacher_identity) if teacher_identity is not None else None),
                anchor_identity=(dict(anchor_identity) if anchor_identity is not None else None),
                judge_identity=(dict(judge_identity) if judge_identity is not None else None),
                judge_cache_manifest=(
                    dict(judge_cache_manifest) if judge_cache_manifest is not None else None
                ),
                teacher_cache_manifest=(
                    dict(teacher_cache_manifest) if teacher_cache_manifest is not None else None
                ),
                completed_evaluation_keys=tuple(sorted(set(completed_evaluation_keys))),
                source_revision=source_revision,
            )
            (temporary / "manifest.json").write_text(
                f"{manifest.model_dump_json()}\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        pointer = {"checkpoint": destination.name, "global_step": step}
        self._write_pointer("latest.json", pointer)
        if is_best:
            self._write_pointer("best.json", pointer)
        return destination

    def restore(
        self,
        trainer: SingleGPUTrainer,
        checkpoint: str | Path | None = None,
        *,
        expected_model_identity: Mapping[str, Any],
        expected_tokenizer_fingerprint: str,
        expected_dataset_fingerprint: str,
        expected_benchmark_fingerprints: Mapping[str, str] | None = None,
    ) -> RestoredCheckpoint:
        """Validate immutable fields before mutating model, optimizer, or RNG state."""
        torch = _torch()
        path = self.resolve_checkpoint(checkpoint)
        manifest = CheckpointManifest.model_validate_json(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        expected = {
            "configuration_fingerprint": self.configuration_fingerprint,
            "resolved_configuration": self.resolved_configuration,
            "model_identity": json.loads(json.dumps(expected_model_identity, allow_nan=False)),
            "tokenizer_fingerprint": expected_tokenizer_fingerprint,
            "dataset_fingerprint": expected_dataset_fingerprint,
            "benchmark_fingerprints": dict(expected_benchmark_fingerprints or {}),
        }
        actual = {
            "configuration_fingerprint": manifest.configuration_fingerprint,
            "resolved_configuration": manifest.resolved_configuration,
            "model_identity": manifest.model_identity,
            "tokenizer_fingerprint": manifest.tokenizer_fingerprint,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "benchmark_fingerprints": manifest.benchmark_fingerprints,
        }
        differences = field_differences(expected, actual)
        if differences:
            rendered = "; ".join(
                f"{name}: expected={values[0]!r}, found={values[1]!r}"
                for name, values in sorted(differences.items())
            )
            raise ValueError(f"incompatible checkpoint fields: {rendered}")
        trainable_state = torch.load(
            path / "student_adapter.pt",
            map_location="cpu",
            weights_only=True,
        )
        if set(trainable_state) != set(trainer.trainable_parameters):
            raise ValueError("checkpoint trainable parameter names do not match the student")
        for name, parameter in trainer.trainable_parameters.items():
            saved = trainable_state[name]
            if saved.shape != parameter.shape or saved.dtype != parameter.dtype:
                raise ValueError(f"checkpoint trainable parameter shape/dtype differs for {name!r}")
        training_state = torch.load(
            path / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if set(training_state) != {
            "optimizer",
            "scheduler",
            "scaler",
            "trainer",
            "rng",
            "controller",
        }:
            raise ValueError("checkpoint training state has unknown or missing fields")
        with torch.no_grad():
            for name, parameter in trainer.trainable_parameters.items():
                parameter.copy_(trainable_state[name].to(parameter.device))
        trainer.optimizer.load_state_dict(training_state["optimizer"])
        move_optimizer_state_to_parameter_devices(trainer.optimizer)
        trainer.scheduler.load_state_dict(training_state["scheduler"])
        trainer.scaler.load_state_dict(training_state["scaler"])
        trainer.state = TrainerState.from_dict(training_state["trainer"])
        if trainer.state.global_step != manifest.global_step:
            raise ValueError("trainer and manifest global steps disagree")
        restore_rng_state(training_state["rng"])
        return RestoredCheckpoint(
            path=path,
            manifest=manifest,
            controller_state=training_state["controller"],
        )

    def resolve_checkpoint(self, checkpoint: str | Path | None = None) -> Path:
        """Resolve an explicit checkpoint or the atomically maintained latest pointer."""
        if checkpoint is None:
            pointer_path = self.run_directory / "latest.json"
            if not pointer_path.is_file():
                raise FileNotFoundError("latest checkpoint metadata does not exist")
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if set(pointer) != {"checkpoint", "global_step"}:
                raise ValueError("latest checkpoint metadata is invalid")
            path = self.run_directory / str(pointer["checkpoint"])
        else:
            path = Path(checkpoint)
            if not path.is_absolute():
                path = self.run_directory / path
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
        return path

    def _write_pointer(self, name: str, payload: dict[str, Any]) -> None:
        path = self.run_directory / name
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            f"{json.dumps(payload, sort_keys=True, allow_nan=False)}\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, CPU, and all available CUDA RNG streams."""
    torch = _torch()
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams only after checkpoint compatibility validates."""
    torch = _torch()
    if set(state) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG state has unknown or missing fields")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def field_differences(
    expected: Any,
    actual: Any,
    *,
    prefix: str = "",
) -> dict[str, tuple[Any, Any]]:
    """Return leaf-level immutable configuration/fingerprint differences."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: dict[str, tuple[Any, Any]] = {}
        keys = set(expected) | set(actual)
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences[name] = ("<absent>", actual[key])
            elif key not in actual:
                differences[name] = (expected[key], "<absent>")
            else:
                differences.update(field_differences(expected[key], actual[key], prefix=name))
        return differences
    if expected != actual:
        return {prefix or "value": (expected, actual)}
    return {}


def move_optimizer_state_to_parameter_devices(optimizer: Any) -> None:
    """Move CPU-restored optimizer tensors beside their owning CUDA parameters."""
    torch = _torch()
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor):
            continue
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(parameter.device)


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training checkpoints") from exc


__all__ = [
    "CheckpointManifest",
    "RestoredCheckpoint",
    "TrainingCheckpointManager",
    "capture_rng_state",
    "field_differences",
    "move_optimizer_state_to_parameter_devices",
    "restore_rng_state",
]
