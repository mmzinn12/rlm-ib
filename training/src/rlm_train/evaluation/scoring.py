"""Benchmark scoring contract with private verifier data kept out of generation."""

from __future__ import annotations

from typing import Any, Protocol

from rlm_train.datasets.records import DatasetRecord


class Scorer(Protocol):
    def score(self, record: DatasetRecord, final_answer: str) -> tuple[float, dict[str, Any]]: ...


__all__ = ["Scorer"]
