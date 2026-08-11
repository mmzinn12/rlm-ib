"""Objective-agnostic training and evaluation datasets."""

from rlm_train.datasets.adapters.hotpotqa import HotpotQADataset
from rlm_train.datasets.adapters.jsonl import JSONLDataset
from rlm_train.datasets.build import build_dataset
from rlm_train.datasets.protocol import Dataset
from rlm_train.datasets.records import DatasetRecord, require_question_context

__all__ = [
    "Dataset",
    "DatasetRecord",
    "HotpotQADataset",
    "JSONLDataset",
    "build_dataset",
    "require_question_context",
]
