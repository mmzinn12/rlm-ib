"""Exact-token characterization tests for the extracted selection package."""

from __future__ import annotations

import pytest

from rlm_train.settings import TokenScope
from rlm_train.token_selection import (
    choose_tokens,
    find_text_regions,
    match_character_range,
    selection_for_schema_v1,
)
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    AnnotationRecord,
    DecisionRole,
    ExecutionNode,
    ExecutionRecord,
    GenerationTokens,
    NodeRole,
    TaskPartition,
)


def character_generation(
    generation_id: str,
    node_id: str,
    text: str,
    *,
    owner: str = "student:one",
) -> GenerationTokens:
    return GenerationTokens(
        generation_id=generation_id,
        node_id=node_id,
        policy_owner=owner,
        text=text,
        prompt_token_ids=(9001,),
        token_ids=tuple(range(100, 100 + len(text))),
        token_offsets=tuple((index, index + 1) for index in range(len(text))),
    )


def representative_attempt() -> AnnotatedRollout:
    root = character_generation(
        "root-generation",
        "root",
        "root prose\n```repl\nvalue = llm_query('useful question')\n```",
    )
    child = character_generation("child-generation", "child", "plain child response")
    foreign = character_generation(
        "foreign-generation",
        "foreign",
        "foreign response",
        owner="student:other",
    )
    root_spans = find_text_regions(
        root,
        node_role=NodeRole.ROOT,
        default_decision_role=DecisionRole.REASONING,
    )
    helper = next(span for span in root_spans if span.decision_role is DecisionRole.HELPER_QUESTION)
    overlapping_helper = helper.model_copy(update={"span_id": f"{helper.span_id}/overlap"})
    spans = (
        *root_spans,
        overlapping_helper,
        *find_text_regions(
            child,
            node_role=NodeRole.PLAIN_SUBCALL,
            default_decision_role=DecisionRole.SUBCALL_RESPONSE,
        ),
        *find_text_regions(
            foreign,
            node_role=NodeRole.PLAIN_SUBCALL,
            default_decision_role=DecisionRole.SUBCALL_RESPONSE,
        ),
    )
    return AnnotatedRollout(
        rollout_id="attempt-1",
        mode="training",
        task=TaskPartition(task_id="task-1", public={"question": "question"}),
        policy={"policy_owner": "student:one"},
        execution=ExecutionRecord(
            root_node_id="root",
            nodes=(
                ExecutionNode(
                    node_id="root",
                    role=NodeRole.ROOT,
                    depth=0,
                    policy_owner="student:one",
                    prompt="question",
                ),
                ExecutionNode(
                    node_id="child",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    policy_owner="student:one",
                    prompt="useful question",
                ),
                ExecutionNode(
                    node_id="foreign",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    policy_owner="student:other",
                    prompt="foreign question",
                ),
            ),
            events=(),
        ),
        annotations=AnnotationRecord(
            generations=(root, child, foreign),
            semantic_spans=spans,
        ),
    )


def selected_text(attempt: AnnotatedRollout, scope: TokenScope) -> str:
    result = choose_tokens(
        attempt,
        training_method="sdpo",
        included_text=scope,
        student_id="student:one",
    )
    generations = {
        generation.generation_id: generation for generation in attempt.annotations.generations
    }
    return "".join(
        generations[item.generation_id].text[position]
        for item in result.selection.generations
        for position in item.positions
    )


def test_match_character_range_uses_contained_exact_offsets() -> None:
    offsets = ((0, 2), (2, 4), (4, 6))

    assert match_character_range(offsets, 0, 4) == (0, 2)
    assert match_character_range(offsets, 1, 6) == (1, 3)
    with pytest.raises(ValueError, match="fully contains no sampled token"):
        match_character_range(offsets, 1, 2)
    with pytest.raises(ValueError, match="outside"):
        match_character_range(offsets, 0, 7)


def test_match_character_range_supports_zero_width_multibyte_offsets() -> None:
    offsets = ((0, 0), (0, 1), (1, 2))

    assert match_character_range(offsets, 0, 1) == (1, 2)
    assert match_character_range(offsets, 0, 2) == (1, 3)


def test_text_regions_are_deterministic_and_structural() -> None:
    generation = representative_attempt().annotations.generations[0]

    first = find_text_regions(
        generation,
        node_role=NodeRole.ROOT,
        default_decision_role=DecisionRole.REASONING,
    )
    second = find_text_regions(
        generation,
        node_role=NodeRole.ROOT,
        default_decision_role=DecisionRole.REASONING,
    )

    assert first == second
    assert [span.span_id for span in first] == [
        f"{generation.generation_id}/span/{index:04d}" for index in range(len(first))
    ]
    helper = next(span for span in first if span.decision_role is DecisionRole.HELPER_QUESTION)
    assert generation.text[helper.char_start : helper.char_end] == "useful question"
    assert generation.token_ids[helper.token_start : helper.token_end] == helper.token_ids


def test_each_scope_selects_exact_student_owned_positions_without_duplicates() -> None:
    attempt = representative_attempt()
    results = {
        scope: choose_tokens(
            attempt,
            training_method=scope.value,
            included_text=scope,
            student_id="student:one",
        )
        for scope in TokenScope
    }

    assert selected_text(attempt, TokenScope.HELPER_QUESTIONS) == "useful question"
    assert selected_text(attempt, TokenScope.SUBCALL_NATURAL_LANGUAGE) == "plain child response"
    natural = selected_text(attempt, TokenScope.NATURAL_LANGUAGE)
    assert "root prose" in natural
    assert "useful question" in natural
    assert "plain child response" in natural
    assert "llm_query" not in natural
    assert "foreign response" not in natural
    assert selected_text(attempt, TokenScope.ALL_STUDENT_TOKENS) == (
        attempt.annotations.generations[0].text + attempt.annotations.generations[1].text
    )
    for result in results.values():
        for generation in result.selection.generations:
            assert generation.positions == tuple(sorted(set(generation.positions)))
            assert generation.generation_id != "foreign-generation"


def test_methods_receive_independent_overlapping_selections_and_empty_is_explicit() -> None:
    attempt = representative_attempt()
    selections = tuple(
        choose_tokens(
            attempt,
            training_method=method,
            included_text=TokenScope.NATURAL_LANGUAGE,
            student_id="student:one",
        ).selection
        for method in ("sdpo", "grpo", "gram")
    )

    assert {selection.training_method for selection in selections} == {"sdpo", "grpo", "gram"}
    positions = {
        tuple(item.positions for item in selection.generations) for selection in selections
    }
    assert len(positions) == 1

    empty_attempt = attempt.model_copy(
        update={"annotations": attempt.annotations.model_copy(update={"semantic_spans": ()})}
    )
    empty = choose_tokens(
        empty_attempt,
        training_method="sdpo",
        included_text=TokenScope.HELPER_QUESTIONS,
        student_id="student:one",
    )
    assert empty.selection.generations == ()
    assert empty.selection.active_token_count == 0
    assert all(not any(mask) for mask in empty.masks.values())


def test_schema_v1_conversion_preserves_attempt_generation_node_and_ids() -> None:
    attempt = representative_attempt()
    result = choose_tokens(
        attempt,
        training_method="sdpo",
        included_text=TokenScope.HELPER_QUESTIONS,
        student_id="student:one",
    )

    durable = selection_for_schema_v1(
        result.selection,
        attempt,
        included_text=TokenScope.HELPER_QUESTIONS,
        student_id="student:one",
    )

    assert durable == result.durable
    assert durable.objective == "sdpo"
    assert durable.policy_owner == "student:one"
    assert {item.node_id for item in durable.ranges} == {"root"}
    generation = attempt.annotations.generations[0]
    assert (
        "".join(
            generation.text[position]
            for item in durable.ranges
            for position in range(item.token_start, item.token_end)
        )
        == "useful question"
    )
