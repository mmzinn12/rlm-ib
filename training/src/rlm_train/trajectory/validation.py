"""Validation helpers for canonical annotated rollouts."""

from rlm_train.trajectory.schema import AnnotatedRollout


def validate_annotated_rollout(rollout: AnnotatedRollout) -> None:
    AnnotatedRollout.model_validate(rollout.model_dump(mode="python"))


__all__ = ["validate_annotated_rollout"]
