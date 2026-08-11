"""Records and helpers for scoring exact sampled continuation tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenPredictions:
    token_ids: tuple[int, ...]
    logprobs: Any | None = None
    logits: Any | None = None
    hidden_states: Any | None = None


def score_continuation_logits(
    model: Any,
    *,
    prompt_token_ids: tuple[int, ...],
    continuation_token_ids: tuple[int, ...],
    with_gradients: bool,
    positions: tuple[int, ...] | None = None,
) -> Any:
    """Score exact continuation IDs, retaining gradients only when requested."""
    from rlm_train.models.transformers_runtime import score_continuation_logits as score

    return score(
        model,
        prompt_token_ids=prompt_token_ids,
        continuation_token_ids=continuation_token_ids,
        require_grad=with_gradients,
        positions=positions,
    )


def continuation_logprobs(
    model: Any,
    *,
    prompt_token_ids: tuple[int, ...],
    continuation_token_ids: tuple[int, ...],
    with_gradients: bool,
    positions: tuple[int, ...] | None = None,
) -> Any:
    from rlm_train.models.transformers_runtime import continuation_logprobs as score

    return score(
        model,
        prompt_token_ids=prompt_token_ids,
        continuation_token_ids=continuation_token_ids,
        require_grad=with_gradients,
        positions=positions,
    )


__all__ = ["TokenPredictions", "continuation_logprobs", "score_continuation_logits"]
