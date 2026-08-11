"""Record canonical RLM events as an immutable annotated attempt."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from rlm.core.events import (
    ExecutionEvent,
    HelperQuestionGenerated,
    InvocationCompleted,
    InvocationFailed,
    InvocationStarted,
    PlainSubcallStarted,
    RecursiveSubcallStarted,
    StudentGenerationCompleted,
    SubcallCompleted,
)

from rlm_train.attempts.attempt_records import (
    AnnotatedAttempt,
    AnnotationRecord,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    GenerationTokens,
    TaskPartition,
)
from rlm_train.token_selection.text_regions import find_text_regions
from rlm_train.trajectory.schema import DecisionRole, NodeRole


def private_fingerprint(value: Any | None) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def generations_from_events(events: tuple[ExecutionEvent, ...]) -> tuple[GenerationTokens, ...]:
    """Preserve exact prompt, sampled-token IDs, and offsets from execution events."""
    generations: list[GenerationTokens] = []
    for event in events:
        if isinstance(event, StudentGenerationCompleted):
            if event.policy_owner is None:
                raise ValueError("student generation is missing policy-owner identity")
            generations.append(
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
            generations.append(
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
    return tuple(generations)


class AttemptRecorder:
    """Thread-safe observer that stores source events and creates a durable attempt."""

    def __init__(
        self,
        *,
        task_id: str,
        public_task: dict[str, Any],
        private_reference: Any | None,
        student: dict[str, Any],
        mode: str,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.task = TaskPartition(
            task_id=task_id,
            public=public_task,
            private_reference_fingerprint=private_fingerprint(private_reference),
        )
        # Schema version 1 calls this record section ``policy``.
        self.student = dict(student)
        self.mode = mode
        self.provenance = dict(provenance or {})
        self._events: list[ExecutionEvent] = []
        self._lock = threading.Lock()

    def observe(self, event: ExecutionEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        with self._lock:
            return tuple(sorted(self._events, key=lambda event: event.sequence_number))

    def create_attempt(self, *, result: dict[str, Any] | None = None) -> AnnotatedAttempt:
        events = self.events
        if not events:
            raise ValueError("cannot create an attempt without execution events")
        nodes: dict[str, ExecutionNode] = {}
        roles: dict[str, NodeRole] = {}
        helper_questions: dict[str, HelperQuestionGenerated] = {}
        edges: dict[str, ExecutionEdge] = {}
        for event in events:
            if isinstance(event, InvocationStarted):
                role = NodeRole(event.node_role)
                roles[event.invocation_id] = role
                nodes[event.invocation_id] = ExecutionNode(
                    node_id=event.invocation_id,
                    parent_id=event.parent_invocation_id,
                    role=role,
                    depth=event.depth,
                    policy_owner=event.policy_owner,
                    source_model=event.source_model,
                    prompt=event.prompt,
                )
            elif isinstance(event, HelperQuestionGenerated):
                helper_questions[event.subcall_id] = event
            elif isinstance(event, PlainSubcallStarted):
                roles[event.subcall_id] = NodeRole.PLAIN_SUBCALL
                nodes[event.subcall_id] = ExecutionNode(
                    node_id=event.subcall_id,
                    parent_id=event.invocation_id,
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=nodes[event.invocation_id].depth + 1,
                    policy_owner=event.policy_owner,
                    source_model=event.source_model,
                    prompt=event.prompt,
                )
            elif isinstance(event, RecursiveSubcallStarted):
                question = helper_questions.get(event.subcall_id)
                roles[event.child_invocation_id] = NodeRole.RECURSIVE_SUBCALL
                nodes[event.child_invocation_id] = ExecutionNode(
                    node_id=event.child_invocation_id,
                    parent_id=event.invocation_id,
                    role=NodeRole.RECURSIVE_SUBCALL,
                    depth=nodes[event.invocation_id].depth + 1,
                    policy_owner=event.policy_owner,
                    source_model=event.source_model,
                    prompt=event.prompt,
                    failed=True,
                )
                edges[event.subcall_id] = ExecutionEdge(
                    edge_id=event.subcall_id,
                    parent_id=event.invocation_id,
                    child_id=event.child_invocation_id,
                    kind="recursive",
                    question=event.prompt if question is None else question.question,
                )
            elif isinstance(event, SubcallCompleted) and event.subcall_kind == "plain":
                node = nodes[event.subcall_id]
                nodes[event.subcall_id] = node.model_copy(
                    update={"result": event.response, "failed": event.error is not None}
                )
                question = helper_questions[event.subcall_id]
                edges[event.subcall_id] = ExecutionEdge(
                    edge_id=event.subcall_id,
                    parent_id=event.invocation_id,
                    child_id=event.subcall_id,
                    kind="plain",
                    question=question.question,
                )
            elif isinstance(event, InvocationCompleted):
                node = nodes[event.invocation_id]
                nodes[event.invocation_id] = node.model_copy(update={"result": event.result})
            elif isinstance(event, InvocationFailed):
                node = nodes[event.invocation_id]
                nodes[event.invocation_id] = node.model_copy(update={"failed": True})

        root_nodes = [node for node in nodes.values() if node.parent_id is None]
        if len(root_nodes) != 1:
            raise ValueError("attempt must contain exactly one root invocation")
        generations = generations_from_events(events)
        semantic_spans = []
        generation_decisions = {
            event.generation_id: DecisionRole(event.decision_role)
            for event in events
            if isinstance(event, StudentGenerationCompleted)
        }
        for generation in generations:
            role = roles[generation.node_id]
            decision = generation_decisions.get(
                generation.generation_id,
                DecisionRole.SUBCALL_RESPONSE
                if role is not NodeRole.ROOT
                else DecisionRole.REASONING,
            )
            semantic_spans.extend(
                find_text_regions(
                    generation,
                    node_role=role,
                    default_decision_role=decision,
                )
            )
        return AnnotatedAttempt(
            # Persisted schema version 1 intentionally retains ``rollout_id``.
            rollout_id=events[0].rollout_id,
            mode=self.mode,
            task=self.task,
            policy=self.student,
            execution=ExecutionRecord(
                root_node_id=root_nodes[0].node_id,
                nodes=tuple(sorted(nodes.values(), key=lambda node: node.node_id)),
                edges=tuple(sorted(edges.values(), key=lambda edge: edge.edge_id)),
                events=tuple(event.to_dict() for event in events),
            ),
            annotations=AnnotationRecord(
                generations=generations,
                semantic_spans=tuple(semantic_spans),
            ),
            result=result or {},
            provenance=self.provenance,
        )


__all__ = ["AttemptRecorder", "generations_from_events", "private_fingerprint"]
