"""Structured response formats accepted from feedback judges."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rlm_train.feedback.feedback_records import RubricFeedback


class InformationSignificance(StrEnum):
    HARMFUL = "harmful"
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InformationLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceQuality(StrEnum):
    POOR = "poor"
    MIXED = "mixed"
    GOOD = "good"


class CategoricalJudgeAssessment(BaseModel):
    """Enum-only output for models that are unreliable at bounded numeric ratings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    significance: InformationSignificance
    novelty: InformationLevel
    uncertainty_reduction: InformationLevel
    evidence_quality: EvidenceQuality
    redundant: bool
    misleading: bool
    diagnostic: str
    information_revealed: tuple[str, ...]
    rubric: RubricFeedback

    def normalized_content(self) -> dict[str, object]:
        significance_scores = {
            InformationSignificance.HARMFUL: -1.0,
            InformationSignificance.NONE: 0.0,
            InformationSignificance.LOW: 0.25,
            InformationSignificance.MEDIUM: 0.6,
            InformationSignificance.HIGH: 1.0,
        }
        level_scores = {
            InformationLevel.NONE: 0.0,
            InformationLevel.LOW: 0.25,
            InformationLevel.MEDIUM: 0.6,
            InformationLevel.HIGH: 1.0,
        }
        evidence_scores = {
            EvidenceQuality.POOR: 0.0,
            EvidenceQuality.MIXED: 0.5,
            EvidenceQuality.GOOD: 1.0,
        }
        return {
            "judge_mode": "categorical",
            "categories": self.model_dump(mode="json"),
            "information_significance": significance_scores[self.significance],
            "novelty": level_scores[self.novelty],
            "uncertainty_reduction": level_scores[self.uncertainty_reduction],
            "evidence_quality": evidence_scores[self.evidence_quality],
            "redundant": self.redundant,
            "misleading": self.misleading,
            "diagnostic": self.diagnostic,
            "information_revealed": list(self.information_revealed),
            "rubric": self.rubric.model_dump(mode="json"),
        }


class FullJudgeAssessment(BaseModel):
    """Bounded numeric output for models that reliably follow the full schema."""

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


__all__ = [
    "CategoricalJudgeAssessment",
    "EvidenceQuality",
    "FullJudgeAssessment",
    "InformationLevel",
    "InformationSignificance",
]
