"""Checkpoint persistence and output-directory ownership tests."""

from __future__ import annotations

import json

import pytest

from rlm_train.artifacts.checkpoints import (
    LATEST_CHECKPOINT_FILENAME,
    TRAINING_STATE_FILENAME,
    TransformersCheckpointWriter,
    resolve_checkpoint_path,
)
from rlm_train.artifacts.run_directory import prepare_training_output
from rlm_train.engine.state import TrainerState

FINGERPRINT = "a" * 64


def test_new_training_output_must_be_empty(tmp_path):
    output = tmp_path / "run"
    assert prepare_training_output(output, resume_checkpoint=None) == output.resolve()
    (output / "metrics.jsonl").write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        prepare_training_output(output, resume_checkpoint=None)


def test_resume_checkpoint_must_belong_to_output_directory(tmp_path):
    output = tmp_path / "run"
    checkpoint = output / "checkpoints" / "step-00000001"
    checkpoint.mkdir(parents=True)

    assert prepare_training_output(output, resume_checkpoint=checkpoint) == output.resolve()
    outside = tmp_path / "other" / "step-00000001"
    outside.mkdir(parents=True)
    with pytest.raises(ValueError, match="must belong"):
        prepare_training_output(output, resume_checkpoint=outside)


def test_checkpoint_round_trip_and_retention(tmp_path):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW((parameter,), lr=0.1)

    class Policy:
        def save_pretrained(self, destination):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model-marker.json").write_text("{}\n", encoding="utf-8")

    writer = TransformersCheckpointWriter(
        tmp_path,
        policy=Policy(),
        optimizer=optimizer,
        scheduler=None,
        checkpoint_interval=1,
        retain_checkpoints=1,
    )
    first = writer.write(
        TrainerState(
            optimizer_step=1,
            examples_seen=4,
            run_spec_fingerprint=FINGERPRINT,
        ),
        final=False,
    )
    second = writer.write(
        TrainerState(
            optimizer_step=2,
            examples_seen=8,
            run_spec_fingerprint=FINGERPRINT,
        ),
        final=True,
    )

    assert first is not None and not first.exists()
    assert second is not None and (second / TRAINING_STATE_FILENAME).exists()
    manifest = tmp_path / LATEST_CHECKPOINT_FILENAME
    assert resolve_checkpoint_path(manifest) == second.resolve()
    assert json.loads(manifest.read_text())["optimizer_step"] == 2
    restored = writer.restore_training_state(second, run_spec_fingerprint=FINGERPRINT)
    assert restored.optimizer_step == 2
    assert restored.examples_seen == 8


def test_checkpoint_restore_rejects_different_run_spec(tmp_path):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)

    class Policy:
        def save_pretrained(self, destination):
            destination.mkdir(parents=True, exist_ok=True)

    writer = TransformersCheckpointWriter(
        tmp_path,
        policy=Policy(),
        optimizer=optimizer,
        scheduler=None,
        checkpoint_interval=None,
        retain_checkpoints=None,
    )
    checkpoint = writer.write(
        TrainerState(optimizer_step=1, run_spec_fingerprint=FINGERPRINT),
        final=True,
    )
    assert checkpoint is not None

    with pytest.raises(ValueError, match="fingerprint"):
        writer.restore_training_state(checkpoint, run_spec_fingerprint="b" * 64)


def test_final_checkpoint_can_be_disabled(tmp_path):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)

    class Policy:
        def save_pretrained(self, destination):
            raise AssertionError(f"unexpected checkpoint write to {destination}")

    writer = TransformersCheckpointWriter(
        tmp_path,
        policy=Policy(),
        optimizer=optimizer,
        scheduler=None,
        checkpoint_interval=None,
        retain_checkpoints=1,
        save_final_checkpoint=False,
    )

    checkpoint = writer.write(
        TrainerState(optimizer_step=1, run_spec_fingerprint=FINGERPRINT), final=True
    )

    assert checkpoint is None
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / LATEST_CHECKPOINT_FILENAME).exists()
