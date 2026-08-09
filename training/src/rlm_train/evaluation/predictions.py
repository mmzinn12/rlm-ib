"""Whole-file JSONL writer for gradable evaluation predictions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rlm_train.evaluation.records import RecursiveEvaluationRecord


class PredictionsJSONLWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        records: Sequence[RecursiveEvaluationRecord],
        *,
        questions: Mapping[str, str | None] | None = None,
    ) -> Path:
        questions = questions or {}
        with self.path.open("w", encoding="utf-8") as stream:
            for record in records:
                payload = record.model_dump(mode="json")
                question = questions.get(record.record_id)
                if question is not None:
                    payload["question"] = question
                stream.write(json.dumps(payload, sort_keys=True))
                stream.write("\n")
        return self.path


__all__ = ["PredictionsJSONLWriter"]
