"""Fail-fast ownership checks for training output directories."""

from __future__ import annotations

from pathlib import Path


def prepare_training_output(
    output_directory: str | Path, *, resume_checkpoint: str | Path | None
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    if resume_checkpoint is None:
        if output.exists() and not output.is_dir():
            raise FileExistsError(f"training output path is not a directory: {output}")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"training output directory is not empty: {output}; "
                "choose a unique output directory or pass --resume-from"
            )
        output.mkdir(parents=True, exist_ok=True)
        return output

    if not output.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {output}")
    checkpoint = Path(resume_checkpoint).expanduser().resolve()
    expected_parent = output / "checkpoints"
    if checkpoint.parent != expected_parent:
        raise ValueError(f"resume checkpoint must belong to {expected_parent}, got {checkpoint}")
    return output


__all__ = ["prepare_training_output"]
