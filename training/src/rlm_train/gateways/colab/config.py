"""Colab input conversion into the canonical RunSpec."""

from __future__ import annotations

from pathlib import Path

from rlm_train.spec import RunSpec


def load_run_spec(path: str | Path) -> RunSpec:
    return RunSpec.from_file(path)


__all__ = ["load_run_spec"]
