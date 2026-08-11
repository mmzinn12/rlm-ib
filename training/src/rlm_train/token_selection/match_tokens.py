"""Match semantic character ranges to exact sampled-token positions."""

from __future__ import annotations


def match_character_range(
    token_offsets: tuple[tuple[int, int], ...], char_start: int, char_end: int
) -> tuple[int, int]:
    """Return the sampled-token interval fully contained by a character range.

    Offsets are captured during generation.  In particular, zero-width offsets are
    retained because byte-level decoders can emit no character until a later token
    completes a multibyte code point.  Such offsets are valid alignment evidence but
    do not themselves identify a character-bearing token to select.
    """
    if char_start < 0 or char_end <= char_start:
        raise ValueError("character range must be non-empty and ordered")
    if not token_offsets:
        raise ValueError("cannot match a character range without sampled-token offsets")

    previous_start = 0
    previous_end = 0
    for token_start, token_end in token_offsets:
        if token_start < 0 or token_end < token_start:
            raise ValueError("sampled-token offsets must be non-negative and ordered")
        if token_start < previous_start or token_end < previous_end:
            raise ValueError("sampled-token offsets must follow generation order")
        previous_start = token_start
        previous_end = token_end
    if char_end > token_offsets[-1][1]:
        raise ValueError("character range lies outside sampled-token offsets")

    selected = tuple(
        position
        for position, (token_start, token_end) in enumerate(token_offsets)
        if token_start >= char_start and token_end <= char_end and token_end > token_start
    )
    if not selected:
        raise ValueError("character range fully contains no sampled token")
    return selected[0], selected[-1] + 1


__all__ = ["match_character_range"]
