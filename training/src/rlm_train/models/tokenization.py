"""Exact-ID token/character alignment helpers."""

from __future__ import annotations

from collections.abc import Sequence


def token_range_for_characters(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    """Return the minimal half-open token range intersecting a character range."""
    if start < 0 or end <= start:
        raise ValueError("character range must be non-empty and ordered")
    selected = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not selected:
        raise ValueError("character range does not intersect any generated token")
    return selected[0], selected[-1] + 1


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


def validate_exact_alignment(
    text: str, token_ids: Sequence[int], offsets: Sequence[tuple[int, int]]
) -> None:
    if len(token_ids) != len(offsets):
        raise ValueError("token IDs and offsets must have equal length")
    previous_start = 0
    for start, end in offsets:
        if start < previous_start or end < start or end > len(text):
            raise ValueError("token offsets are not ordered within generated text")
        previous_start = start


__all__ = [
    "contained_token_range_for_characters",
    "token_range_for_characters",
    "validate_exact_alignment",
]
