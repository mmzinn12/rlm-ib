"""Public exact-token alignment helpers used by rollout enrichment."""

from rlm_train.models.tokenization import (
    contained_token_range_for_characters,
    token_range_for_characters,
    validate_exact_alignment,
)

__all__ = [
    "contained_token_range_for_characters",
    "token_range_for_characters",
    "validate_exact_alignment",
]
