"""Save and restore student, optimizer, scheduler, and training state."""

from rlm_train.artifacts.checkpoints import (
    LATEST_CHECKPOINT_FILENAME,
    TRAINING_STATE_FILENAME,
    TransformersCheckpointWriter,
    resolve_checkpoint_path,
)

CheckpointWriter = TransformersCheckpointWriter

__all__ = [
    "CheckpointWriter",
    "LATEST_CHECKPOINT_FILENAME",
    "TRAINING_STATE_FILENAME",
    "resolve_checkpoint_path",
]
