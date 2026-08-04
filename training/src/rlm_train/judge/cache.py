"""Build stable judge-cache keys and define feedback-cache implementations.

Purpose:
    Reuse judge results only when the task, trajectory decision, evidence, evaluator,
    rubric, and relevant policy version are identical.
Implementation:
    Canonically ordered JSON is hashed with SHA-256. A protocol permits external cache
    backends, while ``MemoryFeedbackCache`` supplies a process-local implementation.
Inputs:
    Task state, node context, recursion action, child result, evidence, and versions.
Outputs:
    A hexadecimal cache key or cached ``TrajectoryFeedback`` instance.
Example:
    ``cache.put(make_feedback_cache_key(...), feedback)``
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol

from rlm_train.judge.schema import TrajectoryFeedback


def content_digest(payload: dict[str, Any]) -> str:
    """Hash a canonical JSON payload for content-addressed storage."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_feedback_cache_key(
    *,
    task: Any,
    node_context: Any,
    recursion_action: Any,
    child_result: Any,
    evidence_snapshot: Any,
    judge_version: str,
    rubric_version: str,
    policy_version: int | None,
) -> str:
    """Create a content-addressed key for one trajectory-feedback evaluation.

    Args:
        task: Original task or prompt representation.
        node_context: Context visible before the evaluated decision.
        recursion_action: Generated call, query, or routing action.
        child_result: Information returned by the child invocation.
        evidence_snapshot: Evidence available to the judge at evaluation time.
        judge_version: Identifier for the evaluator prompt/model implementation.
        rubric_version: Identifier for the feedback rubric.
        policy_version: Optional version of the policy that generated the trajectory.

    Returns:
        A deterministic SHA-256 hexadecimal digest.

    Example:
        ``key = make_feedback_cache_key(task="q", node_context=[], recursion_action="call", child_result="a", evidence_snapshot=None, judge_version="j1", rubric_version="r1", policy_version=3)``
    """
    payload = {
        "task": task,
        "node_context": node_context,
        "recursion_action": recursion_action,
        "child_result": child_result,
        "evidence_snapshot": evidence_snapshot,
        "judge_version": judge_version,
        "rubric_version": rubric_version,
        "policy_version": policy_version,
    }
    return content_digest(payload)


def make_trajectory_feedback_cache_key(
    *,
    task: dict[str, Any],
    trajectory: dict[str, Any],
    privileged_context_fingerprint: str | None,
    judge_version: str,
    rubric_version: str,
) -> str:
    """Create a content-addressed key for one complete trajectory evaluation.

    The raw privileged payload is deliberately excluded. Its fingerprint still makes
    cache reuse sensitive to every privileged-context change.

    Args:
        task: Public task payload safe for persistence.
        trajectory: Complete serialized trajectory tree.
        privileged_context_fingerprint: Optional digest from the judge-only context.
        judge_version: Structured judge implementation/model version.
        rubric_version: Prompt and rubric version.

    Returns:
        A deterministic SHA-256 hexadecimal digest.
    """
    return content_digest(
        {
            "task": task,
            "trajectory": trajectory,
            "privileged_context_fingerprint": privileged_context_fingerprint,
            "judge_version": judge_version,
            "rubric_version": rubric_version,
        }
    )


class FeedbackCache(Protocol):
    """Define storage operations required by a feedback-cache backend."""

    def get(self, key: str) -> TrajectoryFeedback | None:
        """Return cached feedback for ``key``, or ``None`` when absent."""
        ...

    def put(self, key: str, feedback: TrajectoryFeedback) -> None:
        """Store validated feedback under ``key`` and return ``None``."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Return content keys and backend identity without cached feedback payloads."""
        ...


class MemoryFeedbackCache:
    """Store trajectory feedback in a process-local dictionary.

    This implementation is intended for tests and single-process experiments. It is
    not persistent and does not coordinate across workers.

    Example:
        ``cache = MemoryFeedbackCache(); cache.put("key", feedback)``
    """

    _items: dict[str, TrajectoryFeedback]

    def __init__(self) -> None:
        """Initialize an empty feedback map."""
        self._items = {}

    def get(self, key: str) -> TrajectoryFeedback | None:
        """Look up feedback by its content-addressed key.

        Args:
            key: Cache key produced by ``make_feedback_cache_key``.

        Returns:
            Stored feedback, or ``None`` if no entry exists.
        """
        return self._items.get(key)

    def put(self, key: str, feedback: TrajectoryFeedback) -> None:
        """Insert or replace one cache entry.

        Args:
            key: Content-addressed cache key.
            feedback: Validated judge output to store.

        Returns:
            ``None``.
        """
        self._items[key] = feedback

    def manifest(self) -> dict[str, Any]:
        """Return safe process-local cache provenance."""
        return {
            "backend": "memory",
            "count": len(self._items),
            "keys": sorted(self._items),
        }


class SQLiteFeedbackCache:
    """Persist content-addressed feedback in a small SQLite database.

    A new SQLite connection is used for each operation, allowing the cache object to be
    shared across worker threads without retaining thread-bound connection state.

    Example:
        ``cache = SQLiteFeedbackCache("judge-feedback.sqlite3")``
    """

    def __init__(self, path: str | Path) -> None:
        """Create the cache table at ``path`` when it does not already exist."""
        if not str(path).strip():
            raise ValueError("feedback cache path must not be blank")
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("persistent feedback cache requires a filesystem path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectory_feedback (
                    cache_key TEXT PRIMARY KEY,
                    feedback_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def connect(self) -> sqlite3.Connection:
        """Open one bounded-wait SQLite connection."""
        return sqlite3.connect(self.path, timeout=30.0)

    def get(self, key: str) -> TrajectoryFeedback | None:
        """Load and revalidate cached feedback for ``key``."""
        if not key:
            raise ValueError("feedback cache key must not be empty")
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT feedback_json FROM trajectory_feedback WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return TrajectoryFeedback.model_validate_json(row[0])

    def put(self, key: str, feedback: TrajectoryFeedback) -> None:
        """Atomically insert or replace one validated feedback value."""
        if not key:
            raise ValueError("feedback cache key must not be empty")
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO trajectory_feedback(cache_key, feedback_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET feedback_json = excluded.feedback_json
                """,
                (key, feedback.model_dump_json()),
            )
            connection.commit()

    def manifest(self) -> dict[str, Any]:
        """Return persistent cache keys without duplicating private judge context."""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT cache_key FROM trajectory_feedback ORDER BY cache_key"
            ).fetchall()
        return {
            "backend": "sqlite",
            "path": str(self.path.resolve()),
            "count": len(rows),
            "keys": [str(row[0]) for row in rows],
        }


__all__ = [
    "FeedbackCache",
    "MemoryFeedbackCache",
    "SQLiteFeedbackCache",
    "make_feedback_cache_key",
    "make_trajectory_feedback_cache_key",
]
