"""Atomic Transformers checkpoints for training, resumption, and evaluation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rlm_train.engine.state import TrainerState

TRAINING_STATE_FILENAME = "training-state.pt"
LATEST_CHECKPOINT_FILENAME = "latest-checkpoint.json"


class TransformersCheckpointWriter:
    """Persist model and optimizer state under a run's checkpoint directory."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        policy: Any,
        optimizer: Any,
        scheduler: Any | None,
        checkpoint_interval: int | None,
        retain_checkpoints: int | None,
        save_final_checkpoint: bool = True,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.checkpoint_directory = self.output_directory / "checkpoints"
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_interval = checkpoint_interval
        self.retain_checkpoints = retain_checkpoints
        self.save_final_checkpoint = save_final_checkpoint

    def write(self, state: TrainerState, *, final: bool) -> Path | None:
        if final and not self.save_final_checkpoint:
            return None
        if not final and (
            self.checkpoint_interval is None or state.optimizer_step % self.checkpoint_interval != 0
        ):
            return None
        if not hasattr(self.policy, "save_pretrained"):
            raise TypeError("checkpointed policies must implement save_pretrained()")

        import torch

        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        destination = self.checkpoint_directory / f"step-{state.optimizer_step:08d}"
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=self.checkpoint_directory)
        )
        try:
            self.policy.save_pretrained(temporary)
            payload = {
                "trainer_state": state.model_dump(mode="json"),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
            }
            torch.save(payload, temporary / TRAINING_STATE_FILENAME)
            if destination.exists():
                raise FileExistsError(f"checkpoint already exists: {destination}")
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        latest = self.output_directory / LATEST_CHECKPOINT_FILENAME
        latest.write_text(
            json.dumps(
                {
                    "checkpoint_path": str(destination),
                    "optimizer_step": state.optimizer_step,
                    "run_spec_fingerprint": state.run_spec_fingerprint,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._apply_retention(keep=destination)
        return destination

    def restore_training_state(
        self, checkpoint: str | Path, *, run_spec_fingerprint: str
    ) -> TrainerState:
        import torch

        source = resolve_checkpoint_path(checkpoint)
        state_path = source / TRAINING_STATE_FILENAME
        if not state_path.is_file():
            raise FileNotFoundError(f"checkpoint has no training state: {state_path}")
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        state = TrainerState.model_validate(payload["trainer_state"])
        if state.run_spec_fingerprint != run_spec_fingerprint:
            raise ValueError("checkpoint RunSpec fingerprint does not match the requested run")
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler_state = payload["scheduler_state_dict"]
        if scheduler_state is not None:
            if self.scheduler is None:
                raise ValueError(
                    "checkpoint contains scheduler state but this run has no scheduler"
                )
            self.scheduler.load_state_dict(scheduler_state)
        elif self.scheduler is not None:
            raise ValueError("checkpoint has no scheduler state but this run requires one")
        return state

    def _apply_retention(self, *, keep: Path) -> None:
        if self.retain_checkpoints is None:
            return
        checkpoints = sorted(
            path
            for path in self.checkpoint_directory.glob("step-*")
            if path.is_dir() and path != keep
        )
        excess = max(0, len(checkpoints) + 1 - self.retain_checkpoints)
        for path in checkpoints[:excess]:
            shutil.rmtree(path)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve either a checkpoint directory or a latest-checkpoint manifest."""
    source = Path(path).expanduser()
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
        source = Path(payload["checkpoint_path"]).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {source}")
    return source.resolve()


__all__ = [
    "LATEST_CHECKPOINT_FILENAME",
    "TRAINING_STATE_FILENAME",
    "TransformersCheckpointWriter",
    "resolve_checkpoint_path",
]
