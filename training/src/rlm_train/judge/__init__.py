"""Minimal evidence views and scoped judge providers."""

from rlm_train.judge.aggregation import aggregate_overall_assessment
from rlm_train.judge.cache import (
    JudgeCache,
    MemoryJudgeCache,
    SQLiteJudgeCache,
    make_judge_view_cache_key,
)
from rlm_train.judge.prompts import build_judge_instructions, render_judge_view
from rlm_train.judge.protocol import Judge
from rlm_train.judge.providers import DeterministicFakeJudge
from rlm_train.judge.views import JudgeView, build_judge_view

__all__ = [
    "DeterministicFakeJudge",
    "Judge",
    "JudgeCache",
    "JudgeView",
    "MemoryJudgeCache",
    "SQLiteJudgeCache",
    "aggregate_overall_assessment",
    "build_judge_instructions",
    "build_judge_view",
    "make_judge_view_cache_key",
    "render_judge_view",
]
