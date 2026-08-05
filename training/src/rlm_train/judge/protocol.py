"""Structured judge provider contract."""

from __future__ import annotations

from typing import Protocol

from rlm_train.feedback.schema import ScopedAssessment
from rlm_train.judge.views import JudgeView


class Judge(Protocol):
    def assess(self, view: JudgeView) -> ScopedAssessment: ...


__all__ = ["Judge"]
