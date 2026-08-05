"""Content-addressed caches for scoped judge assessments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol

from rlm_train.feedback.schema import ScopedAssessment


def make_judge_view_cache_key(
    *, provider: str, model_revision: str, prompt_version: str, view_fingerprint: str
) -> str:
    """Key an assessment by provider identity and its complete evidence view."""
    if any(not value.strip() for value in (provider, model_revision, prompt_version)):
        raise ValueError("judge cache provenance fields must not be blank")
    if len(view_fingerprint) != 64:
        raise ValueError("judge view fingerprint must be a SHA-256 digest")
    payload = {
        "provider": provider,
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "view_fingerprint": view_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class JudgeCache(Protocol):
    def get(self, key: str) -> ScopedAssessment | None: ...

    def put(self, key: str, assessment: ScopedAssessment) -> None: ...


class MemoryJudgeCache:
    def __init__(self) -> None:
        self.items: dict[str, ScopedAssessment] = {}

    def get(self, key: str) -> ScopedAssessment | None:
        return self.items.get(key)

    def put(self, key: str, assessment: ScopedAssessment) -> None:
        if key != assessment.cache_key:
            raise ValueError("assessment cache key does not match its storage key")
        existing = self.items.get(key)
        if existing is not None and existing != assessment:
            raise ValueError("judge cache key collision")
        self.items[key] = assessment


class SQLiteJudgeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("persistent judge cache requires a filesystem path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scoped_assessment (
                    cache_key TEXT PRIMARY KEY,
                    assessment_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    def get(self, key: str) -> ScopedAssessment | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT assessment_json FROM scoped_assessment WHERE cache_key = ?",
                (key,),
            ).fetchone()
        return None if row is None else ScopedAssessment.model_validate_json(row[0])

    def put(self, key: str, assessment: ScopedAssessment) -> None:
        if key != assessment.cache_key:
            raise ValueError("assessment cache key does not match its storage key")
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO scoped_assessment(cache_key, assessment_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE
                SET assessment_json = excluded.assessment_json
                """,
                (key, assessment.model_dump_json()),
            )
            connection.commit()


__all__ = [
    "JudgeCache",
    "MemoryJudgeCache",
    "SQLiteJudgeCache",
    "make_judge_view_cache_key",
]
