"""Construct judge providers from the immutable run specification."""

from __future__ import annotations

from typing import Any

from rlm_train.judge.cache import JudgeCache
from rlm_train.judge.protocol import Judge
from rlm_train.judge.providers.fake import DeterministicFakeJudge
from rlm_train.judge.providers.openai import OpenAIJudge
from rlm_train.spec.models import JudgeSpec


def build_judge(
    spec: JudgeSpec,
    *,
    client: Any | None = None,
    cache: JudgeCache | None = None,
) -> Judge:
    """Resolve the configured provider and assessment mode."""
    if spec.provider == "fake":
        if client is not None:
            raise ValueError("the fake judge does not accept an API client")
        return DeterministicFakeJudge(
            model_revision=spec.model_revision,
            prompt_version=spec.prompt_version,
        )
    if spec.provider == "openai":
        return OpenAIJudge(spec, client=client, cache=cache)
    raise ValueError(f"unsupported judge provider {spec.provider!r}")


__all__ = ["build_judge"]
