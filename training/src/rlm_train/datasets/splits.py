"""Deterministic dataset split helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rlm_train.datasets.records import DatasetRecord


def deterministic_split(
    records: Sequence[DatasetRecord], *, evaluation_count: int, salt: str
) -> tuple[tuple[DatasetRecord, ...], tuple[DatasetRecord, ...]]:
    if evaluation_count <= 0 or evaluation_count >= len(records):
        raise ValueError("evaluation_count must leave non-empty train and evaluation splits")
    ordered = sorted(
        records,
        key=lambda item: hashlib.sha256(f"{salt}\0{item.record_id}".encode()).digest(),
    )
    evaluation = tuple(ordered[:evaluation_count])
    training = tuple(ordered[evaluation_count:])
    return training, evaluation


__all__ = ["deterministic_split"]
