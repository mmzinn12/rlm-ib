"""Readable end-to-end student training."""

from rlm_train.training.prepare_batch import LossResult, StudentPredictionBatch, TrainingBatch
from rlm_train.training.requirements import TrainingRequirements
from rlm_train.training.training_loop import TrainingLoop, TrainingResult
from rlm_train.training.training_state import TrainingState


def create_training_run(*args, **kwargs):
    from rlm_train.training.create_training_run import create_training_run as create

    return create(*args, **kwargs)


def create_default_training_run(*args, **kwargs):
    from rlm_train.training.create_training_run import create_default_training_run as create

    return create(*args, **kwargs)


def create_training_methods(*args, **kwargs):
    from rlm_train.training.create_training_run import create_training_methods as create

    return create(*args, **kwargs)


__all__ = [
    "LossResult",
    "StudentPredictionBatch",
    "TrainingBatch",
    "TrainingLoop",
    "TrainingRequirements",
    "TrainingResult",
    "TrainingState",
    "create_default_training_run",
    "create_training_methods",
    "create_training_run",
]
