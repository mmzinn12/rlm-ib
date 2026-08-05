"""Visibility-preserving rollout projection."""

from __future__ import annotations

from dataclasses import dataclass

from rlm_train.trajectory.schema import AnnotatedRollout, Visibility


@dataclass(frozen=True)
class RolloutProjectionPolicy:
    allowed_visibilities: frozenset[Visibility] = frozenset({Visibility.PUBLIC})


def project_rollout(rollout: AnnotatedRollout, policy: RolloutProjectionPolicy) -> AnnotatedRollout:
    """Remove semantic annotations above the requested visibility boundary."""
    spans = tuple(
        span
        for span in rollout.annotations.semantic_spans
        if span.visibility in policy.allowed_visibilities
    )
    annotations = rollout.annotations.model_copy(update={"semantic_spans": spans})
    return rollout.model_copy(update={"annotations": annotations})


__all__ = ["RolloutProjectionPolicy", "project_rollout"]
