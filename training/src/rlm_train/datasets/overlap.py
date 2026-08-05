"""Public-task overlap detection independent of objective behavior."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from rlm_train.datasets.records import DatasetRecord


def public_task_overlaps(
    left: Sequence[DatasetRecord], right: Sequence[DatasetRecord]
) -> frozenset[str]:
    def fingerprints(values: Sequence[DatasetRecord]) -> set[str]:
        return {
            hashlib.sha256(
                json.dumps(
                    item.public_task, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()
            for item in values
        }

    return frozenset(fingerprints(left) & fingerprints(right))


__all__ = ["public_task_overlaps"]
