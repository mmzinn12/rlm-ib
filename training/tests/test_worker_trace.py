"""Verify worker-to-proxy trace metadata for single, batched, and recursive calls.

Purpose:
    Protect call-site ordering, batch indices, parent propagation, and helper identity.
Implementation:
    Tests replace the network method with an in-memory capture function, execute REPL
    code through ``Worker``, and inspect emitted proxy payloads.
Inputs:
    Small code strings and root trace contexts.
Outputs:
    Pytest assertions over captured trace metadata and runtime call counts.
Example:
    Run ``pytest training/tests/test_worker_trace.py`` from the repository root.
"""

from typing import Any

from rlm_train.worker import Worker


def test_worker_propagates_parent_and_deterministic_batched_call_metadata():
    worker = Worker(proxy_url="http://unused", rollout_id="run")
    captured: list[dict[str, Any]] = []

    def fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append({"path": path, **payload})
        if path == "llm_query_batched":
            return {"responses": ["one", "two"]}
        return {"response": "single"}

    worker._proxy_post = fake_post

    result = worker.execute(
        "first = llm_query_batched(['a', 'b'])\nsecond = llm_query('c')",
        trace_context={"parent_node_id": "run/root/i000", "call_order_offset": 3},
    )

    batch_contexts = captured[0]["trace_contexts"]
    assert [context["call_order"] for context in batch_contexts] == [3, 3]
    assert [context["batch_index"] for context in batch_contexts] == [0, 1]
    assert captured[1]["trace_context"]["call_order"] == 4
    assert captured[1]["trace_context"]["parent_node_id"] == "run/root/i000"
    assert result["trace_call_count"] == 2


def test_worker_preserves_recursive_helper_kind_in_trace_metadata():
    worker = Worker(proxy_url="http://unused", rollout_id="run")
    captured: list[dict[str, Any]] = []

    def fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append({"path": path, **payload})
        return {"response": "recursive result"}

    worker._proxy_post = fake_post

    worker.execute(
        "result = rlm_query('question')",
        trace_context={"parent_node_id": "run/root/i000"},
    )

    assert captured[0]["trace_context"]["call_kind"] == "rlm_query"
