"""Map response-relative character spans onto exclusive token-level SDPO masks.

Purpose:
    Connect deterministic trajectory segmentation to the tokenizer positions consumed
    by the trainer.
Implementation:
    Each non-empty token offset is assigned to the first overlapping component in the
    fixed precedence order, guaranteeing at most one active mask per token.
Inputs:
    ``DecisionSpan`` objects and response-relative tokenizer offset mappings.
Outputs:
    One boolean mask per ``DecisionKind``, each aligned to the continuation tokens.
Example:
    ``masks = build_exclusive_token_masks(spans, [TokenOffset(0, 4)])``
"""

from __future__ import annotations

from dataclasses import dataclass

from rlm.core.trajectory import CallItemSpan, DecisionKind, DecisionSpan

_TOKEN_PRECEDENCE = [
    DecisionKind.CALL,
    DecisionKind.FINAL,
    DecisionKind.ROUTE,
    DecisionKind.AGGREGATION,
    DecisionKind.NODE,
    DecisionKind.MISSING_CALL,
]


@dataclass(frozen=True)
class TokenOffset:
    """Describe one tokenizer position as a half-open character interval.

    Args:
        start: Inclusive character offset into the generated continuation.
        end: Exclusive character offset; equal offsets represent a special token.

    Raises:
        ValueError: If offsets are negative or reversed.

    Example:
        ``TokenOffset(start=0, end=5)``
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate non-negative, forward token offsets."""
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid token offset")


def build_exclusive_token_masks(
    spans: list[DecisionSpan], token_offsets: list[TokenOffset]
) -> dict[DecisionKind, list[bool]]:
    """Assign every continuation token to at most one SDPO component.

    Args:
        spans: Character spans produced by the response segmenter or compiler.
        token_offsets: Tokenizer offset mappings relative to the same response string.

    Returns:
        A dictionary containing a token-aligned boolean list for every decision kind.
        Zero-width special tokens and tokens outside all spans remain inactive.

    Example:
        ``build_exclusive_token_masks([DecisionSpan(DecisionKind.CALL, 0, 4)], [TokenOffset(0, 4)])``
    """
    masks = {kind: [False] * len(token_offsets) for kind in DecisionKind}
    spans_by_kind = {kind: [span for span in spans if span.kind is kind] for kind in DecisionKind}
    for index, token in enumerate(token_offsets):
        if token.end == token.start:
            continue
        for kind in _TOKEN_PRECEDENCE:
            if any(_overlaps(token, span) for span in spans_by_kind[kind]):
                masks[kind][index] = True
                break
    return masks


def build_question_token_mask(
    question_span: CallItemSpan, token_offsets: list[TokenOffset]
) -> list[bool]:
    """Mask only tokens overlapping one exact question expression.

    The item span excludes its surrounding list delimiters and commas by construction.
    Zero-width special tokens are always inactive. Tokens belonging to sibling
    questions, call answers, routing, and final-answer regions therefore remain false.

    Args:
        question_span: Exact response-relative interval for one scalar or batched item.
        token_offsets: Continuation-token offsets in the same response coordinate space.

    Returns:
        A boolean list aligned to ``token_offsets`` with only overlapping question
        tokens activated.

    Example:
        ``mask = build_question_token_mask(item_span, [TokenOffset(0, 8)])``
    """
    return [
        token.end > token.start
        and token.start < question_span.end
        and question_span.start < token.end
        for token in token_offsets
    ]


def _overlaps(token: TokenOffset, span: DecisionSpan) -> bool:
    """Return whether two half-open character intervals intersect."""
    return token.start < span.end and span.start < token.end
