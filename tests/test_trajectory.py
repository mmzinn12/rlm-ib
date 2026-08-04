"""Verify framework-neutral trajectory types and offset-preserving REPL parsing.

Purpose:
    Protect serialization, tree-reference validation, and response-relative code spans.
Implementation:
    Deterministic unit tests construct small root/child trees and parse representative
    fenced responses without invoking a model or trainer.
Inputs:
    In-memory nodes, spans, serialized dictionaries, and response strings.
Outputs:
    Pytest assertions confirming round trips and expected validation failures.
Example:
    Run ``pytest tests/test_trajectory.py`` from the repository root.
"""

import pytest

from rlm.core.trajectory import (
    CallItemSpan,
    DecisionKind,
    DecisionSpan,
    InvocationKind,
    InvocationNode,
    TrajectoryTree,
)
from rlm.utils.parsing import find_code_blocks_with_spans


def test_trajectory_roundtrip_preserves_edges_and_spans():
    response = "Use llm_query('question')"
    root = InvocationNode(
        node_id="run/root/i000",
        parent_id=None,
        depth=0,
        kind=InvocationKind.ROOT,
        model="model",
        context="context",
        response=response,
        spans=[
            DecisionSpan(
                kind=DecisionKind.CALL,
                start=4,
                end=len(response),
                related_node_id="run/root/i000/c000",
            )
        ],
        call_item_spans=[
            CallItemSpan(
                call_order=0,
                batch_index=None,
                start=14,
                end=len(response) - 1,
                child_node_id="run/root/i000/c000",
            )
        ],
    )
    child = InvocationNode(
        node_id="run/root/i000/c000",
        parent_id=root.node_id,
        depth=1,
        kind=InvocationKind.SUBCALL,
        model="model",
        context="question",
        response="new evidence",
        call_order=0,
    )
    tree = TrajectoryTree(trajectory_id="run", nodes=[root, child])

    restored = TrajectoryTree.from_dict(tree.to_dict())

    assert restored.to_dict() == tree.to_dict()


def test_trajectory_rejects_unknown_parent():
    tree = TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id="child",
                parent_id="missing",
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="model",
                context="question",
            )
        ],
    )

    with pytest.raises(ValueError, match="unknown parent"):
        tree.validate()


def test_code_block_parser_retains_response_offsets():
    response = "before\n```repl\n  value = llm_query('q')  \n```\nafter"

    block = find_code_blocks_with_spans(response)[0]

    assert block.code == "value = llm_query('q')"
    assert response[block.start : block.end] == block.code
    assert response[block.fence_start : block.fence_end].startswith("```repl")


def test_code_block_parser_handles_whitespace_only_blocks():
    block = find_code_blocks_with_spans("```repl\n   \n```")[0]

    assert block.code == ""
    assert block.start == block.end
