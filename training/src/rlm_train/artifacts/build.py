"""Rollout artifact writer construction for training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rlm_train.artifacts.rollout_json import RolloutJSONWriter


def build_artifact_writer(
    output_directory: str | Path,
    *,
    subdir: str = "rollouts",
    mode: Literal["all", "failures", "none"] = "all",
) -> RolloutJSONWriter | None:
    """Create the rollout artifact writer under ``output_directory/subdir``.

    Args:
        output_directory: Run output directory the rollout JSON files are written under.
        subdir: Subdirectory that holds one JSON document per rollout.

    Returns:
        A ``RolloutJSONWriter`` targeting the resolved directory.
    """
    if mode == "none":
        return None
    return RolloutJSONWriter(Path(output_directory) / subdir, mode=mode)


__all__ = ["build_artifact_writer"]
