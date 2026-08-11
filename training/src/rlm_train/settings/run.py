"""Top-level run and runtime settings."""

from rlm_train.spec.run import DatasetRefSpec, RunSpec, RuntimeSpec

DatasetSettings = DatasetRefSpec
RunSettings = RunSpec
RuntimeSettings = RuntimeSpec

__all__ = ["DatasetSettings", "RunSettings", "RuntimeSettings"]
