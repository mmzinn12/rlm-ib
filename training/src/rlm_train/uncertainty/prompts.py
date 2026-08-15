"""Versioned direct-answer uncertainty probes."""

from __future__ import annotations

import json
from typing import Any

DIRECT_ANSWER_PROMPT_VERSION = "direct-answer-v1"


def render_direct_answer_prompt(
    *, question: str, context: Any, helper_information: str, version: str
) -> str:
    if version != DIRECT_ANSWER_PROMPT_VERSION:
        raise ValueError(f"unsupported uncertainty prompt version {version!r}")
    rendered_context = (
        context
        if isinstance(context, str)
        else json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    return (
        f"TASK QUESTION:\n{question}\n\n"
        f"SUPPORTING CONTEXT:\n{rendered_context}\n\n"
        f"AVAILABLE HELPER INFORMATION:\n{helper_information}\n\n"
        "Return only the shortest answer that resolves the task question."
    )


__all__ = ["DIRECT_ANSWER_PROMPT_VERSION", "render_direct_answer_prompt"]
