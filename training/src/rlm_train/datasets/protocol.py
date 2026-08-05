"""Objective-agnostic dataset adapter contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rlm_train.datasets.records import DatasetRecord


class Dataset(Protocol):
    @property
    def identity(self) -> str: ...

    def records(self) -> Sequence[DatasetRecord]: ...


__all__ = ["Dataset"]
