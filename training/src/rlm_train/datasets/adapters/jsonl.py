"""Generic deterministic JSONL dataset adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rlm_train.datasets.records import DatasetRecord


class JSONLDataset:
    def __init__(
        self,
        path: str | Path,
        *,
        prompt_field: str = "prompt",
        target_field: str = "target",
        id_field: str = "id",
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.prompt_field = prompt_field
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
            if self.prompt_field not in payload:
                raise ValueError(f"JSONL line {line_number} is missing {self.prompt_field!r}")
            allowed_fields = {
                self.id_field,
                self.prompt_field,
                self.target_field,
                "metadata",
            }
            unknown = set(payload) - allowed_fields
            if unknown:
                raise ValueError(
                    f"JSONL line {line_number} has unclassified fields: {sorted(unknown)!r}"
                )
            record_id = str(payload.get(self.id_field, line_number))
            public = {"prompt": payload[self.prompt_field]}
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
