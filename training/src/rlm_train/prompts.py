"""Define training-only prompt overlays for traceable recursive decisions.

Purpose:
    Encourage policies to express independent subcall questions as literal lists without
    changing the global inference prompt.
Implementation:
    A constant stores the question-list guidance and an idempotent composition helper
    appends a selected overlay to an experiment's base system prompt.
Inputs:
    A base system-prompt string and an optional overlay string.
Outputs:
    A composed system prompt suitable for ``RLMTrainEnv``.
Example:
    ``prompt = compose_training_system_prompt(RLM_SYSTEM_PROMPT)``
"""

from __future__ import annotations

QUESTION_LIST_PROMPT_OVERLAY = """When unresolved uncertainty would benefit from independent subcalls,
construct an explicit list[str] of independently answerable questions. Each question
should target a distinct uncertainty, and the list should be passed to
llm_query_batched or rlm_query_batched. Do not add redundant questions merely to
increase the number of subcalls."""


def compose_training_system_prompt(
    base_prompt: str,
    *,
    question_list_overlay: str | None = QUESTION_LIST_PROMPT_OVERLAY,
) -> str:
    """Compose an experiment-specific overlay without changing the global RLM prompt.

    Passing ``None`` or an empty string disables the overlay. The operation is
    idempotent so callers can safely reuse an already composed prompt.

    Args:
        base_prompt: Existing inference or custom training system prompt.
        question_list_overlay: Optional experiment guidance to append.

    Returns:
        The right-trimmed base prompt, optionally followed by one copy of the overlay.

    Example:
        ``system_prompt = compose_training_system_prompt(RLM_SYSTEM_PROMPT)``
    """
    overlay = (question_list_overlay or "").strip()
    base = base_prompt.rstrip()
    if not overlay or overlay in base:
        return base
    return f"{base}\n\n{overlay}"


__all__ = ["QUESTION_LIST_PROMPT_OVERLAY", "compose_training_system_prompt"]
