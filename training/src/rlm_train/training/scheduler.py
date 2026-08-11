"""Create the training learning-rate schedule."""

from rlm_train.engine.scheduler import build_training_scheduler

create_scheduler = build_training_scheduler

__all__ = ["create_scheduler"]
