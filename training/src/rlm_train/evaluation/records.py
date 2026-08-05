"""Evaluation result records tied to canonical rollout artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecursiveEvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    sampling_seed: int = Field(ge=0)
    final_answer: str
    score: float
    scoring: dict[str, Any] = Field(default_factory=dict)


__all__ = ["RecursiveEvaluationRecord"]
