"""Identify prose, code, helper questions, and recursive decision regions."""

from __future__ import annotations

import ast

from rlm.utils.parsing import find_code_blocks_with_spans

from rlm_train.token_selection.match_tokens import match_character_range
from rlm_train.trajectory.schema import (
    ContentKind,
    DecisionRole,
    GenerationTokens,
    NodeRole,
    Producer,
    SemanticSpan,
)
from rlm_train.trajectory.segmenter import RLMResponseSegmenter


def find_text_regions(
    generation: GenerationTokens,
    *,
    node_role: NodeRole,
    default_decision_role: DecisionRole,
) -> tuple[SemanticSpan, ...]:
    """Classify prose, generated Python, and helper-question literals structurally."""
    if not generation.token_ids:
        return ()
    text = generation.text
    blocks = find_code_blocks_with_spans(text)
    excluded = [(block.fence_start, block.fence_end) for block in blocks]
    spans: list[SemanticSpan] = []
    ordinal = 0

    def append_span(
        start: int,
        end: int,
        content_kind: ContentKind,
        decision_role: DecisionRole,
    ) -> None:
        nonlocal ordinal
        if end <= start or not text[start:end].strip():
            return
        try:
            token_start, token_end = match_character_range(generation.token_offsets, start, end)
        except ValueError:
            return
        spans.append(
            SemanticSpan(
                span_id=f"{generation.generation_id}/span/{ordinal:04d}",
                generation_id=generation.generation_id,
                node_id=generation.node_id,
                producer=Producer.STUDENT,
                policy_owner=generation.policy_owner,
                node_role=node_role,
                content_kind=content_kind,
                decision_role=decision_role,
                char_start=start,
                char_end=end,
                token_start=token_start,
                token_end=token_end,
                token_ids=generation.token_ids[token_start:token_end],
            )
        )
        ordinal += 1

    cursor = 0
    for start, end in excluded:
        append_span(cursor, start, ContentKind.NATURAL_LANGUAGE, default_decision_role)
        append_span(start, end, ContentKind.CODE, DecisionRole.REASONING)
        cursor = end
    append_span(cursor, len(text), ContentKind.NATURAL_LANGUAGE, default_decision_role)

    if node_role is NodeRole.ROOT:
        segmentation = RLMResponseSegmenter().segment_root(text)
        for item in segmentation.call_item_spans:
            start = item.start
            end = item.end
            source = text[start:end]
            try:
                question = ast.literal_eval(source)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(question, str) or not question:
                continue
            relative_start = source.find(question)
            if relative_start < 0:
                continue
            start += relative_start
            end = start + len(question)
            append_span(
                start,
                end,
                ContentKind.NATURAL_LANGUAGE,
                DecisionRole.HELPER_QUESTION,
            )
    return tuple(spans)


__all__ = ["find_text_regions"]
