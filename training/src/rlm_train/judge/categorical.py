"""Reliable categorical edge assessments and deterministic numeric projection."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


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
    """Enum-only LLM output that avoids unreliable numeric scale generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    significance: InformationSignificance
    novelty: InformationLevel
    uncertainty_reduction: InformationLevel
    evidence_quality: EvidenceQuality
    redundant: bool
    misleading: bool
    diagnostic: str
    information_revealed: tuple[str, ...]

    def normalized_content(self) -> dict[str, object]:
        """Return categories alongside stable bounded values consumed downstream."""
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
        }


__all__ = [
    "CategoricalJudgeAssessment",
    "EvidenceQuality",
    "InformationLevel",
    "InformationSignificance",
]
