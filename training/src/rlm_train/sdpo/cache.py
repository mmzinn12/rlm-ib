"""Cache question-specific teacher targets by every alignment-sensitive input.

Purpose:
    Avoid repeated EMA-teacher forward passes while preventing reuse across questions,
    model versions, tokenizers, or feedback revisions.
Implementation:
    Canonical JSON is content-addressed with SHA-256. A protocol defines cache backends
    and a process-local implementation stores immutable ``TopKTeacherTarget`` values.
Inputs:
    One isolated question example plus teacher, tokenizer, and feedback versions.
Outputs:
    A stable cache key or cached teacher target.
Example:
    ``key = make_question_teacher_cache_key(example, teacher_version=2, tokenizer_fingerprint="tok", feedback_version="rubric-v1")``
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from rlm_train.sdpo.teacher import TopKTeacherTarget, validate_question_teacher_example
from rlm_train.trajectory.compiler import QuestionTrainingExample


def make_question_teacher_cache_key(
    example: QuestionTrainingExample,
    *,
    teacher_version: int,
    tokenizer_fingerprint: str,
    feedback_version: str,
) -> str:
    """Hash complete question identity, content, feedback, and model versions."""
    validate_question_teacher_example(example)
    if teacher_version < 0:
        raise ValueError("teacher_version must be non-negative")
    if not tokenizer_fingerprint.strip() or not feedback_version.strip():
        raise ValueError("tokenizer and feedback versions must not be blank")
    payload = {
        "trajectory_id": example.trajectory_id,
        "parent_node_id": example.parent_node_id,
        "child_node_id": example.child_node_id,
        "call_order": example.call_order,
        "batch_index": example.batch_index,
        "student_context": example.student_context,
        "student_continuation": example.student_continuation,
        "question_span": example.question_span.to_dict(),
        "feedback": example.feedback.model_dump(mode="json"),
        "teacher_version": teacher_version,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "feedback_version": feedback_version,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("teacher cache inputs must be finite JSON-compatible data") from exc
    return hashlib.sha256(encoded).hexdigest()


class TeacherTargetCache(Protocol):
    """Define cache operations used by question teacher scoring."""

    def get(self, key: str) -> TopKTeacherTarget | None:
        """Return a target for ``key``, or ``None`` when absent."""
        ...

    def put(self, key: str, target: TopKTeacherTarget) -> None:
        """Store one validated immutable teacher target."""
        ...


class MemoryTeacherTargetCache:
    """Store teacher targets in one process-local content-addressed map."""

    def __init__(self) -> None:
        """Initialize an empty target map."""
        self.items: dict[str, TopKTeacherTarget] = {}

    def get(self, key: str) -> TopKTeacherTarget | None:
        """Return the cached target for a non-empty key."""
        if not key:
            raise ValueError("teacher cache key must not be empty")
        return self.items.get(key)

    def put(self, key: str, target: TopKTeacherTarget) -> None:
        """Insert or replace one target under a non-empty key."""
        if not key:
            raise ValueError("teacher cache key must not be empty")
        self.items[key] = target


__all__ = [
    "MemoryTeacherTargetCache",
    "TeacherTargetCache",
    "make_question_teacher_cache_key",
]
