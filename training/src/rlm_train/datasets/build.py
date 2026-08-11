"""Dataset construction from a declarative dataset reference.

A ``DatasetRefSpec`` names which adapter loads a dataset and where it lives; ``build_dataset`` is
the single entry point that turns that reference into a concrete ``Dataset`` the trainer and
evaluator can iterate. JSONL and HotpotQA Hub sources share one canonical record boundary.
"""

from __future__ import annotations

from rlm_train.datasets.adapters.hotpotqa import HotpotQADataset
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
        ValueError: If ``ref.adapter`` is unsupported.
    """
    if ref.adapter == "jsonl":
        return JSONLDataset(
            ref.source,
        )
    if ref.adapter == "hotpotqa":
        if ref.subset is None:
            raise ValueError("hotpotqa dataset adapter requires a subset")
        return HotpotQADataset(
            ref.source,
            subset=ref.subset,
            split=ref.split,
            revision=ref.revision,
            max_records=ref.max_records,
        )
    raise ValueError(f"unsupported dataset adapter {ref.adapter!r}; expected 'jsonl' or 'hotpotqa'")


__all__ = ["build_dataset"]
