"""Named benchmark scorers selectable from ExperimentSettings."""

from __future__ import annotations

from typing import Any

from rlm_train.datasets.records import DatasetRecord
from rlm_train.evaluation.scoring import Scorer


def normalize_answer(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("answer", "target", "value", "text"):
            if key in value:
                value = value[key]
                break
    return " ".join(str(value).strip().casefold().split())


class ExactMatchScorer:
    """Score 1.0 when the normalized answer equals the normalized verifier reference."""

    def score(self, record: DatasetRecord, final_answer: str) -> tuple[float, dict[str, Any]]:
        if record.verifier_data is None:
            return 0.0, {"scorer": "exact_match", "reason": "no verifier reference"}
        expected = normalize_answer(record.verifier_data)
        predicted = normalize_answer(final_answer)
        correct = expected == predicted
        return float(correct), {
            "scorer": "exact_match",
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        }


_SCORERS: dict[str, type[Scorer]] = {"exact_match": ExactMatchScorer}


def build_scorer(name: str) -> Scorer:
    if name not in _SCORERS:
        raise ValueError(f"unknown scorer {name!r}; available: {sorted(_SCORERS)!r}")
    return _SCORERS[name]()


__all__ = ["ExactMatchScorer", "build_scorer", "normalize_answer"]
