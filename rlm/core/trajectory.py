"""Define the framework-neutral representation of recursive LM trajectories.

Purpose:
    Provide stable data types shared by inference instrumentation, judge feedback,
    and training without importing a particular trainer or model runtime.
Implementation:
    Dataclasses represent invocation nodes, typed decision spans, and the complete
    trajectory tree. Validation checks node references and response-relative spans,
    while ``to_dict``/``from_dict`` provide a JSON-compatible wire format.
Inputs:
    Policy contexts and responses, node relationships, character spans, model and
    policy identifiers, and optional metadata.
Outputs:
    Validated ``InvocationNode`` and ``TrajectoryTree`` objects or plain dictionaries.
Example:
    ``tree = TrajectoryTree("run", nodes=[root]); tree.validate()``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class InvocationKind(StrEnum):
    """Identify whether an invocation is a root-policy turn or a child subcall."""

    ROOT = "root"
    SUBCALL = "subcall"


class DecisionKind(StrEnum):
    """Name the mutually exclusive policy-decision components used by SDPO masks."""

    ROUTE = "route"
    CALL = "call"
    NODE = "node"
    AGGREGATION = "aggregation"
    FINAL = "final"
    MISSING_CALL = "missing_call"


@dataclass(frozen=True)
class DecisionSpan:
    """Describe one half-open character span over policy-generated response text.

    Args:
        kind: Training component that owns the span.
        start: Inclusive character offset into the node response.
        end: Exclusive character offset into the node response.
        related_node_id: Optional child or otherwise related trajectory node.
        metadata: Extensible span metadata, such as all children of a batched call.

    Raises:
        ValueError: If the offsets do not describe a non-empty forward span.

    Example:
        ``DecisionSpan(DecisionKind.CALL, start=4, end=20)``
    """

    kind: DecisionKind
    start: int
    end: int
    related_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the span is non-empty and ordered."""
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this span to a JSON-compatible dictionary.

        Returns:
            A new dictionary containing primitive values and a copied metadata map.
        """
        return {
            "kind": self.kind.value,
            "start": self.start,
            "end": self.end,
            "related_node_id": self.related_node_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionSpan:
        """Deserialize and validate a decision span.

        Args:
            data: Dictionary produced by :meth:`to_dict` or an equivalent payload.

        Returns:
            A validated ``DecisionSpan``.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If the kind or offsets are invalid.
        """
        return cls(
            kind=DecisionKind(data["kind"]),
            start=int(data["start"]),
            end=int(data["end"]),
            related_node_id=data.get("related_node_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CallItemSpan:
    """Address one question argument within a traced helper call.

    Unlike :class:`DecisionSpan`, which assigns the complete call expression to the
    broad ``CALL`` component, this span covers only one statically addressable question
    expression.  ``batch_index`` is ``None`` for scalar helpers and the zero-based list
    or tuple position for batched helpers.

    Args:
        call_order: Zero-based supported helper-call order in source order.
        batch_index: Item position for batched helpers, or ``None`` for scalar helpers.
        start: Inclusive character offset into the parent response.
        end: Exclusive character offset into the parent response.
        child_node_id: Runtime child bound to this item, when the call executed.

    Raises:
        ValueError: If call coordinates or source offsets are negative, empty, or
            reversed.

    Example:
        ``CallItemSpan(call_order=1, batch_index=0, start=20, end=36)``
    """

    call_order: int
    batch_index: int | None
    start: int
    end: int
    child_node_id: str | None = None

    def __post_init__(self) -> None:
        """Validate source offsets and runtime call coordinates."""
        if self.call_order < 0:
            raise ValueError("call item order must be non-negative")
        if self.batch_index is not None and self.batch_index < 0:
            raise ValueError("call item batch index must be non-negative")
        if self.start < 0:
            raise ValueError("call item span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("call item span end must be greater than its start")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this call item to JSON-compatible primitives.

        Returns:
            A new dictionary containing source coordinates and optional child ID.
        """
        return {
            "call_order": self.call_order,
            "batch_index": self.batch_index,
            "start": self.start,
            "end": self.end,
            "child_node_id": self.child_node_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallItemSpan:
        """Deserialize and validate a question/call-item span.

        Args:
            data: Mapping produced by :meth:`to_dict` or an equivalent wire payload.

        Returns:
            A validated immutable ``CallItemSpan``.

        Raises:
            KeyError: If required coordinates are absent.
            ValueError: If coordinate values violate span invariants.
        """
        return cls(
            call_order=int(data["call_order"]),
            batch_index=(int(data["batch_index"]) if data.get("batch_index") is not None else None),
            start=int(data["start"]),
            end=int(data["end"]),
            child_node_id=data.get("child_node_id"),
        )


@dataclass
class InvocationNode:
    """Represent one policy invocation and its local decisions in an RLM rollout.

    Args:
        node_id: Stable trajectory-local identifier.
        parent_id: Parent invocation identifier, or ``None`` for the first root.
        depth: Recursion depth; root turns use zero and subcalls use a positive value.
        kind: Root or subcall invocation classification.
        model: Model identifier used for this invocation.
        context: Prompt or message context visible to the invocation.
        response: Policy-generated continuation.
        spans: Response-relative training decision spans.
        call_item_spans: Individually addressable question expressions within calls.
        call_order: Zero-based call-site order within the parent response.
        batch_index: Position within a batched call, when applicable.
        consumed_node_ids: Child results visible to this invocation.
        policy_version: Optional policy checkpoint/version used for generation.
        metadata: Extensible invocation metadata.

    Raises:
        ValueError: If node identity, depth, or root/subcall invariants are invalid.

    Example:
        ``InvocationNode("run/root/i000", None, 0, InvocationKind.ROOT, "m", [])``
    """

    node_id: str
    parent_id: str | None
    depth: int
    kind: InvocationKind
    model: str
    context: Any
    response: str = ""
    spans: list[DecisionSpan] = field(default_factory=list)
    call_item_spans: list[CallItemSpan] = field(default_factory=list)
    call_order: int | None = None
    batch_index: int | None = None
    consumed_node_ids: list[str] = field(default_factory=list)
    policy_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate node identity and root/subcall depth invariants."""
        if not self.node_id:
            raise ValueError("node_id must not be empty")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.parent_id == self.node_id:
            raise ValueError("a node cannot be its own parent")
        if self.kind is InvocationKind.ROOT and self.depth != 0:
            raise ValueError("root nodes must have depth 0")
        if self.kind is InvocationKind.SUBCALL:
            if self.parent_id is None:
                raise ValueError("subcall nodes require a parent")
            if self.depth == 0:
                raise ValueError("subcall nodes must have positive depth")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node and its spans to a JSON-compatible dictionary.

        Returns:
            A dictionary suitable for persistence or transport.
        """
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "kind": self.kind.value,
            "model": self.model,
            "context": self.context,
            "response": self.response,
            "spans": [span.to_dict() for span in self.spans],
            "call_item_spans": [span.to_dict() for span in self.call_item_spans],
            "call_order": self.call_order,
            "batch_index": self.batch_index,
            "consumed_node_ids": list(self.consumed_node_ids),
            "policy_version": self.policy_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InvocationNode:
        """Deserialize and validate an invocation node.

        Args:
            data: Dictionary containing the serialized node fields.

        Returns:
            A validated ``InvocationNode`` with reconstructed decision spans.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If enum values or node invariants are invalid.
        """
        return cls(
            node_id=str(data["node_id"]),
            parent_id=data.get("parent_id"),
            depth=int(data["depth"]),
            kind=InvocationKind(data["kind"]),
            model=str(data["model"]),
            context=data.get("context"),
            response=str(data.get("response") or ""),
            spans=[DecisionSpan.from_dict(span) for span in data.get("spans") or []],
            call_item_spans=[
                CallItemSpan.from_dict(span) for span in data.get("call_item_spans") or []
            ],
            call_order=data.get("call_order"),
            batch_index=data.get("batch_index"),
            consumed_node_ids=list(data.get("consumed_node_ids") or []),
            policy_version=data.get("policy_version"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TrajectoryTree:
    """Collect all node-addressable invocations belonging to one rollout.

    Args:
        trajectory_id: Stable identifier shared by every node in the rollout.
        nodes: Root and subcall invocations in deterministic storage order.
        metadata: Rollout-level metadata, such as task or sampler versions.

    Example:
        ``TrajectoryTree("run", [root, child]).validate()``
    """

    trajectory_id: str
    nodes: list[InvocationNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate tree identity, edges, consumed nodes, and span references.

        Returns:
            ``None`` after all invariants have been checked.

        Raises:
            ValueError: If the trajectory ID is empty, node IDs collide, an edge
                references an unknown node, or a span exceeds its response.
        """
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("trajectory node IDs must be unique")
        known = set(node_ids)
        nodes_by_id = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            if node.parent_id is not None and node.parent_id not in known:
                raise ValueError(
                    f"node {node.node_id!r} references unknown parent {node.parent_id!r}"
                )
            missing_consumed = set(node.consumed_node_ids) - known
            if missing_consumed:
                raise ValueError(
                    f"node {node.node_id!r} consumes unknown nodes {sorted(missing_consumed)!r}"
                )
            for span in node.spans:
                if span.end > len(node.response):
                    raise ValueError(
                        f"span {span.start}:{span.end} exceeds response for node {node.node_id!r}"
                    )
                if span.related_node_id is not None and span.related_node_id not in known:
                    raise ValueError(
                        f"span on node {node.node_id!r} references unknown related node "
                        f"{span.related_node_id!r}"
                    )
                related_node_ids = set(span.metadata.get("related_node_ids") or [])
                missing_related = related_node_ids - known
                if missing_related:
                    raise ValueError(
                        f"span on node {node.node_id!r} references unknown related nodes "
                        f"{sorted(missing_related)!r}"
                    )
            item_coordinates: set[tuple[int, int | None]] = set()
            for item_span in node.call_item_spans:
                if item_span.end > len(node.response):
                    raise ValueError(
                        f"call item span {item_span.start}:{item_span.end} exceeds response "
                        f"for node {node.node_id!r}"
                    )
                coordinates = (item_span.call_order, item_span.batch_index)
                if coordinates in item_coordinates:
                    raise ValueError(
                        f"node {node.node_id!r} has duplicate call item coordinates {coordinates!r}"
                    )
                item_coordinates.add(coordinates)
                if item_span.child_node_id is not None and item_span.child_node_id not in known:
                    raise ValueError(
                        f"call item span on node {node.node_id!r} references unknown child "
                        f"{item_span.child_node_id!r}"
                    )
                if item_span.child_node_id is not None:
                    child = nodes_by_id[item_span.child_node_id]
                    if child.kind is not InvocationKind.SUBCALL or child.parent_id != node.node_id:
                        raise ValueError(
                            "call item span must reference a subcall belonging to its node"
                        )
                    if (
                        child.call_order != item_span.call_order
                        or child.batch_index != item_span.batch_index
                    ):
                        raise ValueError("call item span coordinates must match its bound child")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete trajectory to a JSON-compatible dictionary.

        Returns:
            A dictionary containing the trajectory ID, serialized nodes, and metadata.
        """
        return {
            "trajectory_id": self.trajectory_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryTree:
        """Reconstruct and validate a trajectory from its wire representation.

        Args:
            data: Dictionary produced by :meth:`to_dict` or an equivalent payload.

        Returns:
            A fully validated ``TrajectoryTree``.

        Raises:
            KeyError: If the trajectory ID or required node fields are missing.
            ValueError: If any tree, node, edge, or span invariant fails.
        """
        tree = cls(
            trajectory_id=str(data["trajectory_id"]),
            nodes=[InvocationNode.from_dict(node) for node in data.get("nodes") or []],
            metadata=dict(data.get("metadata") or {}),
        )
        tree.validate()
        return tree


class TraceSink(Protocol):
    """Define the minimal node-recording hook for an instrumentation backend.

    Implementations receive completed ``InvocationNode`` objects and decide how to
    store or forward them. ``TrajectoryRecorder`` is the training-side implementation.
    """

    def record_node(self, node: InvocationNode) -> None:
        """Persist one completed invocation node.

        Args:
            node: Invocation to add to the trace.

        Returns:
            ``None``.
        """
        ...
