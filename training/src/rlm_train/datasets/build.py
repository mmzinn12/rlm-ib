"""Dataset construction from a declarative dataset reference.

A ``DatasetRefSpec`` names which adapter loads a dataset and where it lives; ``build_dataset`` is
the single entry point that turns that reference into a concrete ``Dataset`` the trainer and
evaluator can iterate. Only the JSONL adapter is wired today.
"""

from __future__ import annotations

from rlm_train.datasets.adapters.jsonl import JSONLDataset
from rlm_train.datasets.protocol import Dataset
from rlm_train.spec.run import DatasetRefSpec


def build_dataset(ref: DatasetRefSpec) -> Dataset:
    """Build a concrete dataset from a declarative reference.

    Args:
        ref: Dataset reference selecting the adapter and source to load.

    Returns:
        A ``Dataset`` backed by the referenced source.

    Raises:
        ValueError: If ``ref.adapter`` is anything other than the wired ``"jsonl"`` adapter.
    """
    if ref.adapter != "jsonl":
        raise ValueError(f"unsupported dataset adapter {ref.adapter!r}; only 'jsonl' is wired")
    return JSONLDataset(ref.source)


__all__ = ["build_dataset"]
