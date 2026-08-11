"""Exact-ID token/character alignment helpers."""

from __future__ import annotations

from collections.abc import Sequence


def contained_token_range_for_characters(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Return only tokens fully contained by a structural character range.

    Conservative containment prevents a boundary token that also contains generated
    Python or call syntax from entering a natural-language-only objective.
    """
    if start < 0 or end <= start:
        raise ValueError("character range must be non-empty and ordered")
    selected = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start >= start and token_end <= end and token_end > token_start
    ]
    if not selected:
        raise ValueError("character range fully contains no generated token")
    return selected[0], selected[-1] + 1
__all__ = ["contained_token_range_for_characters"]
