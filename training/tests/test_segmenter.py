"""Verify deterministic, exclusive response segmentation for the REPL policy.

Purpose:
    Protect call, routing, final, child-reasoning, and aggregation span classification.
Implementation:
    Representative root and child responses are segmented and checked for exact text,
    exclusivity, and exclusion of new REPL scaffolding from aggregation.
Inputs:
    In-memory response strings containing prose and fenced Python code.
Outputs:
    Pytest assertions over ordered ``DecisionSpan`` objects.
Example:
    Run ``pytest training/tests/test_segmenter.py`` from the repository root.
"""

from rlm.core.trajectory import DecisionKind

from rlm_train.trajectory.segmenter import RLMResponseSegmenter


def test_segmenter_extracts_exclusive_route_call_and_final_spans():
    response = """```repl
if uncertainty > 0.5:
    evidence = llm_query(context[10:20])
answer["content"] = evidence
answer["ready"] = True
```"""

    spans = RLMResponseSegmenter().segment_root_response(response)

    kinds = {span.kind for span in spans}
    assert DecisionKind.ROUTE in kinds
    assert DecisionKind.CALL in kinds
    assert DecisionKind.FINAL in kinds
    for index, left in enumerate(spans):
        for right in spans[index + 1 :]:
            assert left.end <= right.start or right.end <= left.start


def test_child_response_is_entire_node_reasoning_span():
    response = "  material new evidence  "

    spans = RLMResponseSegmenter().segment_child_response(response)

    assert len(spans) == 1
    assert spans[0].kind is DecisionKind.NODE
    assert response[spans[0].start : spans[0].end] == "material new evidence"


def test_post_subcall_unclaimed_text_becomes_aggregation():
    response = "The child result conflicts with the baseline."

    spans = RLMResponseSegmenter().segment_root_response(response, has_child_results=True)

    assert len(spans) == 1
    assert spans[0].kind is DecisionKind.AGGREGATION


def test_call_inside_final_assignment_keeps_the_narrow_call_span():
    response = """```repl
answer["content"] = llm_query("resolve uncertainty")
answer["ready"] = True
```"""

    spans = RLMResponseSegmenter().segment_root_response(response)

    call_spans = [span for span in spans if span.kind is DecisionKind.CALL]
    assert len(call_spans) == 1
    assert response[call_spans[0].start : call_spans[0].end] == 'llm_query("resolve uncertainty")'


def test_aggregation_does_not_claim_new_repl_scaffolding():
    response = """Synthesize the returned evidence.
```repl
next_result = llm_query("follow-up")
```"""

    spans = RLMResponseSegmenter().segment_root_response(response, has_child_results=True)

    aggregation = [span for span in spans if span.kind is DecisionKind.AGGREGATION]
    assert len(aggregation) == 1
    assert response[aggregation[0].start : aggregation[0].end] == (
        "Synthesize the returned evidence."
    )


def test_segmenter_extracts_each_literal_question_from_a_named_list():
    response = """```repl
questions = [
    "What supports mechanism A?",
    "Does the trial exclude population B?",
]
results = llm_query_batched(questions)
```"""

    result = RLMResponseSegmenter().segment_root(response)

    assert [response[span.start : span.end] for span in result.call_item_spans] == [
        '"What supports mechanism A?"',
        '"Does the trial exclude population B?"',
    ]
    assert [span.batch_index for span in result.call_item_spans] == [0, 1]
    assert {span.call_order for span in result.call_item_spans} == {0}
    assert result.question_item_count == 2
    assert result.unaddressable_question_item_count == 0


def test_segmenter_marks_dynamic_batches_unaddressable_without_guessing_spans():
    response = """```repl
questions = [build_question(item) for item in context]
results = rlm_query_batched(questions)
```"""

    result = RLMResponseSegmenter().segment_root(response)

    assert result.call_item_spans == []
    assert result.question_item_count == 1
    assert result.unaddressable_question_item_count == 1
    call = next(span for span in result.spans if span.kind is DecisionKind.CALL)
    assert call.metadata["unaddressable_question_item_count"] == 1


def test_segmenter_does_not_address_literal_questions_inside_dynamic_control_flow():
    response = """```repl
for topic in context:
    result = llm_query("inspect topic")
```"""

    result = RLMResponseSegmenter().segment_root(response)

    assert result.call_item_spans == []
    assert result.question_item_count == 1
    assert result.unaddressable_question_item_count == 1
