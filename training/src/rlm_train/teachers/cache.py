"""Teacher-target cache keyed by complete target provenance."""

from __future__ import annotations

from rlm_train.teachers.targets import TeacherTarget


class MemoryTeacherTargetCache:
    def __init__(self) -> None:
        self._values: dict[str, TeacherTarget] = {}

    def get(self, key: str) -> TeacherTarget | None:
        return self._values.get(key)

    def put(self, target: TeacherTarget) -> None:
        existing = self._values.get(target.target_id)
        if existing is not None and existing != target:
            raise ValueError("teacher target cache key collision")
        self._values[target.target_id] = target


__all__ = ["MemoryTeacherTargetCache"]
