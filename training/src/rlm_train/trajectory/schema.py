"""Canonical immutable annotated-rollout record."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Producer(StrEnum):
    STUDENT = "student"
    ENVIRONMENT = "environment"
    JUDGE = "judge"
    TEACHER = "teacher"
    VERIFIER = "verifier"


class NodeRole(StrEnum):
    ROOT = "root"
    PLAIN_SUBCALL = "plain_subcall"
    RECURSIVE_SUBCALL = "recursive_subcall"


class ContentKind(StrEnum):
    NATURAL_LANGUAGE = "natural_language"
    CODE = "code"
    EXECUTION_OUTPUT = "execution_output"
    CONTROL = "control"


class DecisionRole(StrEnum):
    REASONING = "reasoning"
    HELPER_QUESTION = "helper_question"
    ROUTING = "routing"
    SUBCALL_RESPONSE = "subcall_response"
    AGGREGATION = "aggregation"
    FINAL_ANSWER = "final_answer"


class Visibility(StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    PRIVILEGED = "privileged"


class TaskPartition(ImmutableRecord):
    task_id: str = Field(min_length=1)
    public: dict[str, Any]
    private_reference_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_public_json(self) -> TaskPartition:
        require_json_value("task public data", self.public)
        leaked = find_sensitive_keys(self.public)
        if leaked:
            raise ValueError(f"task public data contains verifier-owned keys: {sorted(leaked)!r}")
        return self


class ExecutionNode(ImmutableRecord):
    node_id: str = Field(min_length=1)
    parent_id: str | None = None
    role: NodeRole
    depth: int = Field(ge=0)
    policy_owner: str | None = None
    source_model: str | None = None
    prompt: Any
    result: str | None = None
    failed: bool = False

    @model_validator(mode="after")
    def validate_prompt_json(self) -> ExecutionNode:
        require_json_value("execution node prompt", self.prompt)
        return self


class ExecutionEdge(ImmutableRecord):
    edge_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    child_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    question: str


class GenerationTokens(ImmutableRecord):
    generation_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    policy_owner: str = Field(min_length=1)
    text: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def validate_alignment(self) -> GenerationTokens:
        if not self.prompt_token_ids or not self.token_ids:
            raise ValueError(
                "canonical training generations require exact prompt and sampled token IDs"
            )
        if len(self.token_ids) != len(self.token_offsets):
            raise ValueError("generation token IDs and offsets must align")
        for start, end in self.token_offsets:
            if start < 0 or end < start or end > len(self.text):
                raise ValueError("generation token offset lies outside generated text")
        return self


class SemanticSpan(ImmutableRecord):
    span_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    producer: Producer
    policy_owner: str | None
    node_role: NodeRole
    content_kind: ContentKind
    decision_role: DecisionRole
    visibility: Visibility = Visibility.PUBLIC
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)
    token_ids: tuple[int, ...]

    @model_validator(mode="after")
    def validate_ranges(self) -> SemanticSpan:
        if self.char_end <= self.char_start or self.token_end <= self.token_start:
            raise ValueError("semantic ranges must be non-empty and ordered")
        if len(self.token_ids) != self.token_end - self.token_start:
            raise ValueError("semantic token IDs must match its token range")
        return self


class SelectedTokenRange(ImmutableRecord):
    generation_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)
    token_ids: tuple[int, ...]
    reason: str = Field(min_length=1)
    source_span_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_range(self) -> SelectedTokenRange:
        if self.token_end <= self.token_start:
            raise ValueError("selected token range must be non-empty")
        if len(self.token_ids) != self.token_end - self.token_start:
            raise ValueError("selected token IDs must match selected range")
        return self


class ObjectiveSelection(ImmutableRecord):
    objective: str = Field(min_length=1)
    token_scope: str = Field(min_length=1)
    policy_owner: str = Field(min_length=1)
    ranges: tuple[SelectedTokenRange, ...]

    @property
    def active_token_count(self) -> int:
        return sum(item.token_end - item.token_start for item in self.ranges)


class ExecutionRecord(ImmutableRecord):
    root_node_id: str = Field(min_length=1)
    nodes: tuple[ExecutionNode, ...]
    edges: tuple[ExecutionEdge, ...] = ()
    events: tuple[dict[str, Any], ...]


class AnnotationRecord(ImmutableRecord):
    generations: tuple[GenerationTokens, ...] = ()
    semantic_spans: tuple[SemanticSpan, ...] = ()
    objective_selections: dict[str, ObjectiveSelection] = Field(default_factory=dict)


class FeedbackRecord(ImmutableRecord):
    environment: dict[str, Any] = Field(default_factory=dict)
    judge_assessments: tuple[dict[str, Any], ...] = ()
    projections: tuple[dict[str, Any], ...] = ()
    overall_assessment: dict[str, Any] = Field(default_factory=dict)
    uncertainty_assessments: tuple[dict[str, Any], ...] = ()


class AnnotatedRollout(ImmutableRecord):
    schema_version: int = 1
    rollout_id: str = Field(min_length=1)
    mode: str = Field(pattern=r"^(training|evaluation)$")
    task: TaskPartition
    policy: dict[str, Any]
    execution: ExecutionRecord
    annotations: AnnotationRecord = Field(default_factory=AnnotationRecord)
    feedback: FeedbackRecord = Field(default_factory=FeedbackRecord)
    teacher_targets: tuple[dict[str, Any], ...] = ()
    objectives: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> AnnotatedRollout:
        if self.schema_version != 1:
            raise ValueError(f"unsupported annotated rollout schema {self.schema_version}")
        node_ids = {node.node_id for node in self.execution.nodes}
        if self.execution.root_node_id not in node_ids:
            raise ValueError("root node is missing from execution nodes")
        for node in self.execution.nodes:
            if node.parent_id is not None and node.parent_id not in node_ids:
                raise ValueError(f"node {node.node_id!r} references unknown parent")
        for edge in self.execution.edges:
            if edge.parent_id not in node_ids or edge.child_id not in node_ids:
                raise ValueError(f"edge {edge.edge_id!r} references an unknown node")
        sequences = [int(event["sequence_number"]) for event in self.execution.events]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("execution events must have unique increasing sequence numbers")
        generation_ids = {item.generation_id for item in self.annotations.generations}
        for span in self.annotations.semantic_spans:
            if span.node_id not in node_ids or span.generation_id not in generation_ids:
                raise ValueError(f"semantic span {span.span_id!r} has an unknown reference")
        for name, value in (
            ("policy", self.policy),
            ("events", self.execution.events),
            ("teacher targets", self.teacher_targets),
            ("objectives", self.objectives),
            ("result", self.result),
            ("provenance", self.provenance),
        ):
            require_json_value(name, value)
        return self

    def canonical_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json(indent=None).encode()).hexdigest()


def require_json_value(name: str, value: Any) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc


def find_sensitive_keys(value: Any) -> set[str]:
    forbidden = {
        "target",
        "target_data",
        "reference_answer",
        "answer_key",
        "verifier_data",
        "private_reference",
    }
    if isinstance(value, dict):
        found = {str(key) for key in value if str(key).lower() in forbidden}
        for item in value.values():
            found.update(find_sensitive_keys(item))
        return found
    if isinstance(value, (list, tuple)):
        return {key for item in value for key in find_sensitive_keys(item)}
    return set()


__all__ = [
    "AnnotatedRollout",
    "AnnotationRecord",
    "ContentKind",
    "DecisionRole",
    "ExecutionEdge",
    "ExecutionNode",
    "ExecutionRecord",
    "FeedbackRecord",
    "GenerationTokens",
    "NodeRole",
    "ObjectiveSelection",
    "Producer",
    "SelectedTokenRange",
    "SemanticSpan",
    "TaskPartition",
    "Visibility",
]
