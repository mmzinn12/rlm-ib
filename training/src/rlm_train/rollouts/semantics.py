"""Structural semantic annotation for canonical RLM generation events."""

from __future__ import annotations

import ast
from collections.abc import Iterable

from rlm.core.events import StudentGenerationCompleted, SubcallCompleted
from rlm.utils.parsing import find_code_blocks_with_spans

from rlm_train.models.tokenization import contained_token_range_for_characters
from rlm_train.trajectory.schema import (
    ContentKind,
    DecisionRole,
    GenerationTokens,
    NodeRole,
    Producer,
    SemanticSpan,
)
from rlm_train.trajectory.segmenter import RLMResponseSegmenter


def annotate_generation(
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
            token_start, token_end = contained_token_range_for_characters(
                generation.token_offsets, start, end
            )
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


def generations_from_events(
    events: Iterable[object], node_roles: dict[str, NodeRole]
) -> tuple[GenerationTokens, ...]:
    values: list[GenerationTokens] = []
    for event in events:
        if isinstance(event, StudentGenerationCompleted):
            if event.policy_owner is None:
                raise ValueError("student generation is missing policy-owner identity")
            values.append(
                GenerationTokens(
                    generation_id=event.generation_id,
                    node_id=event.invocation_id,
                    policy_owner=event.policy_owner,
                    text=event.text,
                    prompt_token_ids=event.prompt_token_ids,
                    token_ids=event.token_ids,
                    token_offsets=event.token_offsets,
                )
            )
        elif isinstance(event, SubcallCompleted) and event.subcall_kind == "plain":
            if event.error is not None:
                continue
            if event.policy_owner is None:
                raise ValueError("plain subcall is missing policy-owner identity")
            values.append(
                GenerationTokens(
                    generation_id=f"{event.subcall_id}/generation",
                    node_id=event.subcall_id,
                    policy_owner=event.policy_owner,
                    text=event.response,
                    prompt_token_ids=event.prompt_token_ids,
                    token_ids=event.token_ids,
                    token_offsets=event.token_offsets,
                )
            )
    return tuple(values)


__all__ = ["annotate_generation", "generations_from_events"]
