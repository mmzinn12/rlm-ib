"""Immutable selected-token records independent of any loss calculation."""

from __future__ import annotations

from dataclasses import dataclass

from rlm_train.settings.token_selection import TokenScope
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    ObjectiveSelection,
    SelectedTokenRange,
)


@dataclass(frozen=True)
class SelectedGenerationTokens:
    generation_id: str
    positions: tuple[int, ...]
    text_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.generation_id or not self.positions:
            raise ValueError("selected generation tokens require an ID and positions")
        if len(set(self.positions)) != len(self.positions) or any(
            position < 0 for position in self.positions
        ):
            raise ValueError("selected token positions must be unique and non-negative")
        if self.positions != tuple(sorted(self.positions)):
            raise ValueError("selected token positions must follow generation order")


@dataclass(frozen=True)
class TokenSelection:
    training_method: str
    attempt_id: str
    generations: tuple[SelectedGenerationTokens, ...]

    def __post_init__(self) -> None:
        if not self.training_method or not self.attempt_id:
            raise ValueError("token selection requires training-method and attempt IDs")
        generation_ids = tuple(item.generation_id for item in self.generations)
        if len(set(generation_ids)) != len(generation_ids):
            raise ValueError("token selection generation IDs must be unique")

    @property
    def active_token_count(self) -> int:
        return sum(len(generation.positions) for generation in self.generations)


@dataclass(frozen=True)
class TokenSelectionResult:
    selection: TokenSelection
    durable: ObjectiveSelection
    masks: dict[str, tuple[bool, ...]]


def selection_for_schema_v1(
    selection: TokenSelection,
    attempt: AnnotatedRollout,
    *,
    included_text: TokenScope,
    student_id: str,
) -> ObjectiveSelection:
    """Convert the runtime selection at the schema-version-1 artifact boundary."""
    if selection.attempt_id != attempt.rollout_id:
        raise ValueError("token selection belongs to a different attempt")
    generations = {
        generation.generation_id: generation for generation in attempt.annotations.generations
    }
    ranges: list[SelectedTokenRange] = []
    for selected_generation in selection.generations:
        generation = generations.get(selected_generation.generation_id)
        if generation is None:
            raise ValueError("token selection references an unknown generation")
        if generation.policy_owner != student_id:
            raise ValueError("token selection references a generation owned by another student")
        if selected_generation.positions[-1] >= len(generation.token_ids):
            raise ValueError("token selection position lies outside its generation")

        start = selected_generation.positions[0]
        previous = start
        intervals: list[tuple[int, int]] = []
        for position in selected_generation.positions[1:]:
            if position != previous + 1:
                intervals.append((start, previous + 1))
                start = position
            previous = position
        intervals.append((start, previous + 1))
        for token_start, token_end in intervals:
            source_span_ids = tuple(
                span.span_id
                for span in attempt.annotations.semantic_spans
                if span.generation_id == generation.generation_id
                and span.policy_owner == student_id
                and span.token_start < token_end
                and span.token_end > token_start
                and span.span_id in selected_generation.text_regions
            )
            ranges.append(
                SelectedTokenRange(
                    generation_id=generation.generation_id,
                    node_id=generation.node_id,
                    token_start=token_start,
                    token_end=token_end,
                    token_ids=generation.token_ids[token_start:token_end],
                    reason=f"{included_text.value} structural selection",
                    source_span_ids=source_span_ids,
                )
            )
    return ObjectiveSelection(
        objective=selection.training_method,
        token_scope=included_text.value,
        policy_owner=student_id,
        ranges=tuple(ranges),
    )


__all__ = [
    "SelectedGenerationTokens",
    "TokenSelection",
    "TokenSelectionResult",
    "selection_for_schema_v1",
]
