"""Bounded numeric structured output for full LLM judge assessments."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rlm_train.feedback.schema import RubricFeedback


class FullJudgeAssessment(BaseModel):
    """Rich numeric assessment retained for models that follow bounded schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    information_significance: float = Field(ge=-1.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    uncertainty_reduction: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    redundant: bool
    misleading: bool
    diagnostic: str
    information_revealed: tuple[str, ...]
    rationale: str
    rubric: RubricFeedback

    def normalized_content(self) -> dict[str, object]:
        return {"judge_mode": "full", **self.model_dump(mode="json")}


__all__ = ["FullJudgeAssessment"]
