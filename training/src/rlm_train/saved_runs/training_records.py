"""Persist immutable annotated attempts."""

from rlm_train.artifacts.rollout_json import RolloutJSONWriter

TrainingRecordWriter = RolloutJSONWriter

__all__ = ["TrainingRecordWriter"]
