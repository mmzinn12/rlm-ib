"""Generic deterministic JSONL dataset adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rlm_train.datasets.records import DatasetRecord


class JSONLDataset:
    """Deterministic dataset backed by a JSON Lines file, one record per non-blank line.

    Each line must be a JSON object with distinct question and context fields and may include an
    id, a target, and a ``metadata`` object; any other key is rejected. The question and evidence
    context become the public task while the target remains verifier-owned. Keeping these fields
    separate matches production RLM inference: the question is visible to the orchestrator and
    only the evidence payload is offloaded into the REPL.

    Attributes:
        path: Filesystem path to the JSONL file (must exist at construction).
        question_field: Line key holding the user question (default ``"question"``).
        context_field: Line key holding the evidence context (default ``"context"``).
        target_field: Line key holding the verifier target, if present (default ``"target"``).
        id_field: Line key holding the record id; falls back to the 1-based line number.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        question_field: str = "question",
        context_field: str = "context",
        target_field: str = "target",
        id_field: str = "id",
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.question_field = question_field
        self.context_field = context_field
        self.target_field = target_field
        self.id_field = id_field
        self._content = self.path.read_bytes()

    @property
    def identity(self) -> str:
        return f"jsonl:{self.path.name}:{hashlib.sha256(self._content).hexdigest()}"

    def records(self) -> tuple[DatasetRecord, ...]:
        values = []
        for line_number, line in enumerate(self._content.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            missing = {self.question_field, self.context_field} - set(payload)
            if missing:
                raise ValueError(
                    f"JSONL line {line_number} must keep question and context separate; "
                    f"missing {sorted(missing)!r}"
                )
            allowed_fields = {
                self.id_field,
                self.question_field,
                self.context_field,
                self.target_field,
                "metadata",
            }
            unknown = set(payload) - allowed_fields
            if unknown:
                raise ValueError(
                    f"JSONL line {line_number} has unclassified fields: {sorted(unknown)!r}"
                )
            record_id = str(payload.get(self.id_field, line_number))
            question = payload[self.question_field]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"JSONL line {line_number} question must be a non-empty string")
            context = payload[self.context_field]
            if context is None:
                raise ValueError(f"JSONL line {line_number} context must not be null")
            public = {"question": question, "context": context}
            verifier = payload.get(self.target_field)
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"JSONL line {line_number} metadata must be an object")
            values.append(
                DatasetRecord(
                    record_id=record_id,
                    public_task=public,
                    verifier_data=verifier,
                    metadata=metadata,
                )
            )
        if not values:
            raise ValueError("JSONL dataset contains no records")
        identifiers = [item.record_id for item in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("JSONL dataset record IDs must be unique")
        return tuple(values)


__all__ = ["JSONLDataset"]
