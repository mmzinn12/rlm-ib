"""Create the configured feedback judge and optional persistent cache."""

from __future__ import annotations

from typing import Any

from rlm_train.judge.cache import JudgeCache, SQLiteJudgeCache
from rlm_train.judge.fake_judge import DeterministicFakeJudge
from rlm_train.judge.judge import FeedbackJudge
from rlm_train.judge.openai_judge import OpenAIJudge
from rlm_train.settings.judge import JudgeSettings


def create_judge(
    settings: JudgeSettings,
    *,
    client: Any | None = None,
    cache: JudgeCache | None = None,
) -> FeedbackJudge:
    """Create a judge from settings, including its configured SQLite cache."""
    if cache is not None and settings.cache_path is not None:
        raise ValueError("pass either a judge cache or cache_path settings, not both")
    resolved_cache = (
        SQLiteJudgeCache(settings.cache_path)
        if cache is None and settings.cache_path is not None
        else cache
    )
    if settings.provider == "fake":
        if client is not None:
            raise ValueError("the fake judge does not accept an API client")
        if resolved_cache is not None:
            raise ValueError("the fake judge does not use an assessment cache")
        return DeterministicFakeJudge(
            model_revision=settings.model_revision,
            prompt_version=settings.prompt_version,
        )
    if settings.provider == "openai":
        return OpenAIJudge(settings, client=client, cache=resolved_cache)
    raise ValueError(f"unsupported judge provider {settings.provider!r}")


__all__ = ["create_judge"]
