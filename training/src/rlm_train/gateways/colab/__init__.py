"""Thin Colab gateway over the public API."""

from rlm_train.api import evaluate, train
from rlm_train.gateways.colab.config import load_run_spec
from rlm_train.gateways.colab.drive import artifact_directory
from rlm_train.gateways.colab.runtime import validate_colab_device

__all__ = [
    "artifact_directory",
    "evaluate",
    "load_run_spec",
    "train",
    "validate_colab_device",
]
