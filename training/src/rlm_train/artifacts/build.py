"""Rollout artifact writer construction for training runs."""

from __future__ import annotations

from pathlib import Path

from rlm_train.artifacts.rollout_json import RolloutJSONWriter


def build_artifact_writer(
    output_directory: str | Path, *, subdir: str = "rollouts"
) -> RolloutJSONWriter:
    """Create the rollout artifact writer under ``output_directory/subdir``.

    Args:
        output_directory: Run output directory the rollout JSON files are written under.
        subdir: Subdirectory that holds one JSON document per rollout.

    Returns:
        A ``RolloutJSONWriter`` targeting the resolved directory.
    """
    return RolloutJSONWriter(Path(output_directory) / subdir)


__all__ = ["build_artifact_writer"]
