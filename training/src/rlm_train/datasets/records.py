"""Dataset records with public task and verifier-owned data separated."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_VERIFIER_KEYS = frozenset({"target", "target_data", "reference", "reference_answer", "answer_key"})


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    public_task: dict[str, Any]
    verifier_data: Any = Field(default=None, repr=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_public_partition(self) -> DatasetRecord:
        for name, payload in (("public_task", self.public_task), ("metadata", self.metadata)):
            leaked = _find_keys(payload, _VERIFIER_KEYS)
            if leaked:
                raise ValueError(f"{name} contains verifier-owned keys: {sorted(leaked)!r}")
        return self

    @property
    def verifier_fingerprint(self) -> str | None:
        if self.verifier_data is None:
            return None
        payload = json.dumps(
            self.verifier_data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def public_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "public_task": json.loads(json.dumps(self.public_task)),
            "metadata": json.loads(json.dumps(self.metadata)),
        }


def _find_keys(value: Any, forbidden: frozenset[str]) -> set[str]:
    if isinstance(value, dict):
        found = {str(key) for key in value if str(key).lower() in forbidden}
        for item in value.values():
            found.update(_find_keys(item, forbidden))
        return found
    if isinstance(value, (list, tuple)):
        return {key for item in value for key in _find_keys(item, forbidden)}
    return set()


__all__ = ["DatasetRecord"]
