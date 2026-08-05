"""Grouped reward normalization for GRPO."""

from __future__ import annotations

import math
from collections.abc import Sequence


def group_relative_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite rewards within a group, including zero-variance groups."""
    if not rewards:
        raise ValueError("advantage calculation requires at least one reward")
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError("group rewards must be finite")
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    if variance == 0.0:
        return (0.0,) * len(rewards)
    scale = math.sqrt(variance)
    return tuple((value - mean) / scale for value in rewards)


__all__ = ["group_relative_advantages"]
