"""Verify stable trajectory IDs and call-span binding in the recorder.

Purpose:
    Protect deterministic identifiers for root, single-call, and batched child nodes.
Implementation:
    A synthetic root response is recorded, completed, connected to three children, and
    snapshotted for inspection.
Inputs:
    In-memory invocation metadata, responses, and call spans.
Outputs:
    Pytest assertions over node IDs and batched ``related_node_ids`` metadata.
Example:
    Run ``pytest training/tests/test_trajectory_recorder.py`` from the repository root.
"""

from rlm.core.trajectory import CallItemSpan, DecisionKind, DecisionSpan, InvocationKind

from rlm_train.trajectory.recorder import TrajectoryRecorder
from rlm_train.trajectory.validation import summarize_question_trace


def test_recorder_assigns_deterministic_root_and_subcall_ids():
    recorder = TrajectoryRecorder("run")
    response = "```repl\na = llm_query('a')\nb = llm_query_batched(['b', 'c'])\n```"
    root_id = recorder.begin_node(
        kind=InvocationKind.ROOT,
        model="model",
        context="context",
        depth=0,
    )
    recorder.complete_node(
        root_id,
        response=response,
        spans=[
            DecisionSpan(kind=DecisionKind.CALL, start=12, end=26),
            DecisionSpan(kind=DecisionKind.CALL, start=31, end=60),
        ],
        call_item_spans=[
            CallItemSpan(call_order=0, batch_index=None, start=22, end=25),
            CallItemSpan(call_order=1, batch_index=0, start=50, end=53),
            CallItemSpan(call_order=1, batch_index=1, start=55, end=58),
        ],
    )

    first = recorder.begin_node(
        kind=InvocationKind.SUBCALL,
        model="model",
        context="a",
        parent_id=root_id,
        depth=1,
        call_order=0,
    )
    batch_one = recorder.begin_node(
        kind=InvocationKind.SUBCALL,
        model="model",
        context="b",
        parent_id=root_id,
        depth=1,
        call_order=1,
        batch_index=0,
    )
    batch_two = recorder.begin_node(
        kind=InvocationKind.SUBCALL,
        model="model",
        context="c",
        parent_id=root_id,
        depth=1,
        call_order=1,
        batch_index=1,
    )
    for node_id in (first, batch_one, batch_two):
        recorder.complete_node(node_id, response="evidence")
    recorder.bind_call_span(root_id, 0, first)
    recorder.bind_call_span(root_id, 1, batch_one)
    recorder.bind_call_span(root_id, 1, batch_two)
    recorder.bind_call_item(root_id, 0, None, first)
    recorder.bind_call_item(root_id, 1, 0, batch_one)
    recorder.bind_call_item(root_id, 1, 1, batch_two)

    tree = recorder.snapshot()

    assert first.endswith("/c000")
    assert batch_one.endswith("/c001/b000")
    assert batch_two.endswith("/c001/b001")
    related = tree.nodes[0].spans[1].metadata["related_node_ids"]
    assert related == [batch_one, batch_two]
    assert [span.child_node_id for span in tree.nodes[0].call_item_spans] == [
        first,
        batch_one,
        batch_two,
    ]
    assert summarize_question_trace(tree, question_feedback_count=3).to_dict() == {
        "question_item_count": 3,
        "bound_question_item_count": 3,
        "unaddressable_question_item_count": 0,
        "question_feedback_count": 3,
    }
