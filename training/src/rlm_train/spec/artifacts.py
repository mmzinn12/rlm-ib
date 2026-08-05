"""Artifact, metric, and checkpoint destination configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rlm_train.spec.models import ImmutableSpec


class ArtifactSpec(ImmutableSpec):
    output_directory: str = Field(default="outputs", min_length=1)
    rollout_json: Literal["all", "failures", "none"] = "all"
    metrics_jsonl: bool = True
    checkpoint_interval: int | None = Field(default=None, gt=0)
    retain_checkpoints: int | None = Field(default=None, gt=0)


__all__ = ["ArtifactSpec"]
