"""Minimal typed evidence views constructed before judge prompt rendering."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlm_train.feedback.schema import FeedbackVisibility
from rlm_train.spec.feedback import AssessmentScope
from rlm_train.spec.objectives import TokenScope
from rlm_train.trajectory.schema import AnnotatedRollout


class JudgeView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    builder_name: str = Field(min_length=1)
    scope: AssessmentScope
    focal_node_ids: tuple[str, ...] = ()
    focal_edge_ids: tuple[str, ...] = ()
    evidence_node_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    upstream_depth: int = Field(ge=0)
    downstream_depth: int = Field(ge=0)
    siblings_included: bool = False
    final_answer_included: bool = False
    verifier_reference_included: bool = False
    visibility: FeedbackVisibility
    allowed_objectives: frozenset[str]
    allowed_token_scopes: frozenset[TokenScope]
    task: dict[str, Any]
    evidence: dict[str, Any]
    verifier_reference: Any | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_boundary(self) -> JudgeView:
        if self.scope is AssessmentScope.CAUSAL_LOCAL:
            if (
                self.downstream_depth
                or self.final_answer_included
                or self.verifier_reference_included
            ):
                raise ValueError(
                    "causal-local view cannot include downstream/final/private evidence"
                )
        if self.scope is AssessmentScope.RETROSPECTIVE_LOCAL and self.verifier_reference_included:
            raise ValueError("retrospective-local view cannot include verifier reference")
        if (
            self.verifier_reference_included
            and self.visibility is not FeedbackVisibility.PRIVILEGED
        ):
            raise ValueError("verifier reference requires privileged visibility")
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


def build_judge_view(
    rollout: AnnotatedRollout,
    *,
    scope: AssessmentScope,
    focal_node_ids: tuple[str, ...] = (),
    focal_edge_ids: tuple[str, ...] = (),
    upstream_depth: int = 1,
    downstream_depth: int = 1,
    include_siblings: bool = False,
    include_final_answer: bool = False,
    verifier_reference: Any | None = None,
    allowed_objectives: frozenset[str] = frozenset(),
    allowed_token_scopes: frozenset[TokenScope] = frozenset(),
    builder_name: str = "minimal-evidence-v1",
) -> JudgeView:
    """Construct an ID-addressed view without relying on prompt instructions for privacy."""
    nodes = {node.node_id: node for node in rollout.execution.nodes}
    edges = {edge.edge_id: edge for edge in rollout.execution.edges}
    if any(node_id not in nodes for node_id in focal_node_ids):
        raise ValueError("judge view references an unknown focal node")
    if any(edge_id not in edges for edge_id in focal_edge_ids):
        raise ValueError("judge view references an unknown focal edge")
    seeds = set(focal_node_ids)
    downstream_roots = {(node_id, 0) for node_id in focal_node_ids}
    for edge_id in focal_edge_ids:
        seeds.add(edges[edge_id].parent_id)
        if scope is not AssessmentScope.CAUSAL_LOCAL:
            seeds.add(edges[edge_id].child_id)
            downstream_roots.add((edges[edge_id].child_id, 1))
    if not seeds:
        raise ValueError("judge view requires at least one focal node or edge")

    included = set(seeds)
    frontier = set(seeds)
    for _ in range(upstream_depth):
        frontier = {nodes[node_id].parent_id for node_id in frontier if nodes[node_id].parent_id}
        included.update(frontier)

    children: dict[str, set[str]] = defaultdict(set)
    for node in nodes.values():
        if node.parent_id is not None:
            children[node.parent_id].add(node.node_id)
    permitted_downstream = downstream_depth if scope is not AssessmentScope.CAUSAL_LOCAL else 0
    queue = deque(sorted(downstream_roots))
    while queue:
        node_id, depth = queue.popleft()
        if depth >= permitted_downstream:
            continue
        for child_id in sorted(children[node_id]):
            included.add(child_id)
            queue.append((child_id, depth + 1))
    if include_siblings:
        for node_id in tuple(included):
            parent_id = nodes[node_id].parent_id
            if parent_id is not None:
                included.update(children[parent_id])

    final_allowed = include_final_answer and scope is AssessmentScope.PRIVILEGED_DIAGNOSTIC
    reference_allowed = (
        verifier_reference is not None and scope is AssessmentScope.PRIVILEGED_DIAGNOSTIC
    )
    visibility = (
        FeedbackVisibility.PRIVILEGED if reference_allowed else FeedbackVisibility.RESTRICTED
    )
    safe_nodes = []
    for node_id in sorted(included):
        node_payload = nodes[node_id].model_dump(mode="json")
        if scope is AssessmentScope.CAUSAL_LOCAL or node_id in seeds:
            node_payload["result"] = None
        if node_id == rollout.execution.root_node_id and not final_allowed:
            node_payload["result"] = None
        safe_nodes.append(node_payload)

    safe_events = []
    for event in rollout.execution.events:
        event_type = event.get("event_type")
        subcall_id = event.get("subcall_id")
        invocation_id = event.get("invocation_id")
        if invocation_id not in included and subcall_id not in focal_edge_ids:
            continue
        if scope is AssessmentScope.PRIVILEGED_DIAGNOSTIC:
            safe_events.append(event)
            continue
        if event_type == "helper_question_generated" and subcall_id in focal_edge_ids:
            safe_events.append(event)
        elif (
            scope is AssessmentScope.RETROSPECTIVE_LOCAL
            and event_type == "subcall_completed"
            and subcall_id in focal_edge_ids
        ):
            safe_events.append(event)
        elif event_type == "invocation_started":
            payload = dict(event)
            if invocation_id != rollout.execution.root_node_id:
                payload["prompt"] = None
            safe_events.append(payload)
    event_payloads = tuple(safe_events)
    return JudgeView(
        builder_name=builder_name,
        scope=scope,
        focal_node_ids=focal_node_ids,
        focal_edge_ids=focal_edge_ids,
        evidence_node_ids=tuple(sorted(included)),
        evidence_event_ids=tuple(str(event["event_id"]) for event in event_payloads),
        upstream_depth=upstream_depth,
        downstream_depth=permitted_downstream,
        siblings_included=include_siblings,
        final_answer_included=final_allowed,
        verifier_reference_included=reference_allowed,
        visibility=visibility,
        allowed_objectives=allowed_objectives,
        allowed_token_scopes=allowed_token_scopes,
        task=rollout.task.public,
        evidence={
            "nodes": safe_nodes,
            "edges": [
                edge.model_dump(mode="json")
                for edge in rollout.execution.edges
                if edge.parent_id in included and edge.child_id in included
            ],
            "events": list(event_payloads),
            "final_answer": rollout.result.get("final_answer") if final_allowed else None,
        },
        verifier_reference=verifier_reference if reference_allowed else None,
    )


__all__ = ["JudgeView", "build_judge_view"]
