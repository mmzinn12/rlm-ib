"""Construct stable, node-addressable trajectory trees during concurrent rollouts.

Purpose:
    Record root turns, single subcalls, and batched subcalls as deterministic nodes that
    can later receive judge feedback.
Implementation:
    An ``RLock`` protects mutable node storage and counters. IDs encode root iteration,
    call-site order, and batch position. Snapshots deep-copy and validate the tree so
    callers cannot mutate recorder state indirectly.
Inputs:
    Invocation metadata at node start, followed by responses and decision spans.
Outputs:
    Stable node IDs and immutable-by-copy ``TrajectoryTree`` snapshots.
Example:
    ``node_id = TrajectoryRecorder("run").begin_node(kind=InvocationKind.ROOT, model="m", context=[], depth=0)``
"""

from __future__ import annotations

import copy
import threading
from collections import defaultdict
from typing import Any

from rlm.core.trajectory import (
    CallItemSpan,
    DecisionKind,
    DecisionSpan,
    InvocationKind,
    InvocationNode,
    TrajectoryTree,
)


class TrajectoryRecorder:
    """Incrementally build one trajectory using thread-safe stable node identifiers.

    Args:
        trajectory_id: Non-empty identifier shared by all generated node IDs.
        metadata: Optional rollout-level metadata copied into every snapshot.

    Raises:
        ValueError: If ``trajectory_id`` is empty.

    Example:
        ``recorder = TrajectoryRecorder("rollout-17")``
    """

    def __init__(self, trajectory_id: str, metadata: dict[str, Any] | None = None):
        """Initialize empty node storage and deterministic ID counters."""
        if not trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        self.trajectory_id = trajectory_id
        self.metadata = dict(metadata or {})
        self._nodes: dict[str, InvocationNode] = {}
        self._node_order: list[str] = []
        self._root_counter = 0
        self._subcall_counters: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def begin_node(
        self,
        *,
        kind: InvocationKind,
        model: str,
        context: Any,
        parent_id: str | None = None,
        depth: int,
        call_order: int | None = None,
        batch_index: int | None = None,
        policy_version: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create an incomplete node and return its stable identifier.

        Args:
            kind: Root or subcall invocation classification.
            model: Model identifier used by the invocation.
            context: Prompt/messages visible to the invocation; deep-copied on input.
            parent_id: Existing parent node for subcalls or chained root turns.
            depth: Recursion depth recorded on the node.
            call_order: Optional explicit zero-based call-site order.
            batch_index: Optional zero-based position within a batched call.
            policy_version: Optional generation-policy version.
            metadata: Optional node-level trace metadata.

        Returns:
            A deterministic node ID. Root IDs end in ``root/iNNN``; child IDs add
            ``cNNN`` and optionally ``bNNN``.

        Raises:
            ValueError: If the parent is unknown, subcall metadata is incomplete,
                indices are negative, or an ID would collide.
        """
        with self._lock:
            if parent_id is not None and parent_id not in self._nodes:
                raise ValueError(f"unknown parent node {parent_id!r}")
            if kind is InvocationKind.ROOT:
                node_id = f"{self.trajectory_id}/root/i{self._root_counter:03d}"
                self._root_counter += 1
            else:
                if parent_id is None:
                    raise ValueError("subcall nodes require a parent_id")
                if call_order is None:
                    index = self._subcall_counters[parent_id]
                else:
                    if call_order < 0:
                        raise ValueError("call_order must be non-negative")
                    index = call_order
                self._subcall_counters[parent_id] = max(
                    self._subcall_counters[parent_id], index + 1
                )
                node_id = f"{parent_id}/c{index:03d}"
                if batch_index is not None:
                    if batch_index < 0:
                        raise ValueError("batch_index must be non-negative")
                    node_id = f"{node_id}/b{batch_index:03d}"
                if node_id in self._nodes:
                    raise ValueError(
                        f"duplicate subcall order {index} for parent node {parent_id!r}"
                    )
            node = InvocationNode(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                kind=kind,
                model=model,
                context=copy.deepcopy(context),
                call_order=call_order,
                batch_index=batch_index,
                policy_version=policy_version,
                metadata=dict(metadata or {}),
            )
            self._nodes[node_id] = node
            self._node_order.append(node_id)
            return node_id

    def complete_node(
        self,
        node_id: str,
        *,
        response: str,
        spans: list[DecisionSpan] | None = None,
        call_item_spans: list[CallItemSpan] | None = None,
        consumed_node_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach a response, spans, consumed nodes, and metadata to an existing node.

        Args:
            node_id: Node returned by :meth:`begin_node`.
            response: Policy-generated continuation.
            spans: Optional response-relative decision spans.
            call_item_spans: Optional exact spans for individual helper questions.
            consumed_node_ids: Child results visible to this invocation.
            metadata: Optional metadata merged into the existing node metadata.

        Returns:
            ``None``.

        Raises:
            ValueError: If ``node_id`` is unknown.
        """
        with self._lock:
            node = self._get_node(node_id)
            node.response = response
            node.spans = list(spans or [])
            node.call_item_spans = list(call_item_spans or [])
            node.consumed_node_ids = list(consumed_node_ids or [])
            if metadata:
                node.metadata.update(metadata)

    def bind_call_span(self, parent_id: str, call_order: int, child_node_id: str) -> None:
        """Bind the Nth call span in a parent response to one generated child.

        Batched children share a call span. Their IDs are accumulated in sorted
        ``related_node_ids`` metadata while ``related_node_id`` points to the first.
        If runtime control flow creates a call without a matching static span, the
        association is retained in parent ``unbound_subcalls`` metadata.

        Args:
            parent_id: Node whose response contains the call expression.
            call_order: Zero-based call span index in response order.
            child_node_id: Existing child produced by that call site.

        Raises:
            ValueError: If the parent or child ID is unknown.
        """
        with self._lock:
            parent = self._get_node(parent_id)
            self._get_node(child_node_id)
            call_spans = [
                span
                for span in parent.spans
                if span.kind is DecisionKind.CALL and span.metadata.get("call_order") == call_order
            ]
            if not call_spans:
                positional_spans = [span for span in parent.spans if span.kind is DecisionKind.CALL]
                if 0 <= call_order < len(positional_spans):
                    call_spans = [positional_spans[call_order]]
            if not call_spans:
                parent.metadata.setdefault("unbound_subcalls", []).append(
                    {"call_order": call_order, "child_node_id": child_node_id}
                )
                return
            target = call_spans[0]
            related_node_ids = list(target.metadata.get("related_node_ids") or [])
            if (
                target.related_node_id is not None
                and target.related_node_id not in related_node_ids
            ):
                related_node_ids.append(target.related_node_id)
            related_node_ids.append(child_node_id)
            related_node_ids = sorted(set(related_node_ids))
            metadata = dict(target.metadata)
            metadata["related_node_ids"] = related_node_ids
            replacement = DecisionSpan(
                kind=target.kind,
                start=target.start,
                end=target.end,
                related_node_id=related_node_ids[0],
                metadata=metadata,
            )
            span_index = parent.spans.index(target)
            parent.spans[span_index] = replacement

    def bind_call_item(
        self,
        parent_node_id: str,
        call_order: int,
        batch_index: int | None,
        child_node_id: str,
    ) -> None:
        """Bind one static question item to the child produced by that runtime call.

        Missing static spans are recorded explicitly rather than approximated. This is
        the expected path for comprehensions, dynamically constructed prompt lists, and
        call sites whose runtime execution order differs from static source order.

        Args:
            parent_node_id: ID of the root node containing the question expression.
            call_order: Zero-based order of the helper call in the root response.
            batch_index: Zero-based item position for a batched call, or ``None`` for
                a scalar call.
            child_node_id: ID of the runtime child that answered this question.

        Returns:
            ``None`` after updating the matching span or recording an explicit unbound
            item in parent metadata when no exact static span exists.

        Raises:
            ValueError: If the child belongs to another parent, the coordinates match
                multiple spans, or the span is already bound to a different child.

        Example:
            ``recorder.bind_call_item(root_id, 0, None, child_id)``
        """
        with self._lock:
            parent = self._get_node(parent_node_id)
            child = self._get_node(child_node_id)
            if child.parent_id != parent_node_id:
                raise ValueError("call item child must belong to the supplied parent")
            matches = [
                (index, span)
                for index, span in enumerate(parent.call_item_spans)
                if span.call_order == call_order and span.batch_index == batch_index
            ]
            if not matches:
                parent.metadata.setdefault("unbound_call_items", []).append(
                    {
                        "call_order": call_order,
                        "batch_index": batch_index,
                        "child_node_id": child_node_id,
                    }
                )
                return
            if len(matches) != 1:
                raise ValueError("call item coordinates must identify exactly one span")
            index, target = matches[0]
            if target.child_node_id is not None and target.child_node_id != child_node_id:
                raise ValueError("call item span is already bound to a different child")
            parent.call_item_spans[index] = CallItemSpan(
                call_order=target.call_order,
                batch_index=target.batch_index,
                start=target.start,
                end=target.end,
                child_node_id=child_node_id,
            )

    def record_node(self, node: InvocationNode) -> None:
        """Insert a preconstructed invocation node.

        Args:
            node: Completed node to deep-copy into recorder storage.

        Raises:
            ValueError: If its ID already exists or its parent is unknown.
        """
        with self._lock:
            if node.node_id in self._nodes:
                raise ValueError(f"duplicate node ID {node.node_id!r}")
            if node.parent_id is not None and node.parent_id not in self._nodes:
                raise ValueError(f"unknown parent node {node.parent_id!r}")
            self._nodes[node.node_id] = copy.deepcopy(node)
            self._node_order.append(node.node_id)

    def snapshot(self) -> TrajectoryTree:
        """Return a deterministic, deeply copied, validated trajectory snapshot.

        Returns:
            A ``TrajectoryTree`` whose node order is stable by node ID.

        Raises:
            ValueError: If accumulated node, edge, or span invariants are invalid.
        """
        with self._lock:
            tree = TrajectoryTree(
                trajectory_id=self.trajectory_id,
                nodes=[copy.deepcopy(self._nodes[node_id]) for node_id in sorted(self._node_order)],
                metadata=copy.deepcopy(self.metadata),
            )
            tree.validate()
            return tree

    def _get_node(self, node_id: str) -> InvocationNode:
        """Return mutable internal node storage or raise a public validation error."""
        try:
            return self._nodes[node_id]
        except KeyError as e:
            raise ValueError(f"unknown node {node_id!r}") from e
