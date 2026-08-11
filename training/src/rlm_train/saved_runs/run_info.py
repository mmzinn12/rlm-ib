"""Training-run provenance and output-directory preparation."""

from rlm_train.artifacts.provenance import RunProvenance
from rlm_train.artifacts.run_directory import prepare_training_output

RunInfo = RunProvenance

__all__ = ["RunInfo", "prepare_training_output"]
