"""Immutable events emitted by the canonical recursive execution engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, ClassVar


class EventProducer(StrEnum):
    """Component that produced an execution value."""

    STUDENT = "student"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


@dataclass(frozen=True, kw_only=True)
class RLMEvent:
    """Common identity and provenance carried by every execution event."""

    rollout_id: str
    event_id: str
    invocation_id: str
    parent_invocation_id: str | None
    sequence_number: int
    producer: EventProducer
    policy_owner: str | None = None
    source_model: str | None = None

    event_type: ClassVar[str] = "rlm_event"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible event payload."""
        payload = asdict(self)
        payload["producer"] = self.producer.value
        payload["event_type"] = self.event_type
        return payload


@dataclass(frozen=True, kw_only=True)
class InvocationStarted(RLMEvent):
    event_type: ClassVar[str] = "invocation_started"
    prompt: Any
    depth: int
    node_role: str


@dataclass(frozen=True, kw_only=True)
class StudentGenerationStarted(RLMEvent):
    event_type: ClassVar[str] = "student_generation_started"
    generation_id: str
    prompt: Any
    decision_role: str = "reasoning"


@dataclass(frozen=True, kw_only=True)
class StudentGenerationCompleted(RLMEvent):
    event_type: ClassVar[str] = "student_generation_completed"
    generation_id: str
    text: str
    prompt_token_ids: tuple[int, ...] = ()
    token_ids: tuple[int, ...] = ()
    token_offsets: tuple[tuple[int, int], ...] = ()
    prompt_token_count: int | None = None
    decision_role: str = "reasoning"


@dataclass(frozen=True, kw_only=True)
class CodeExecutionStarted(RLMEvent):
    event_type: ClassVar[str] = "code_execution_started"
    code: str
    block_index: int


@dataclass(frozen=True, kw_only=True)
class CodeExecutionCompleted(RLMEvent):
    event_type: ClassVar[str] = "code_execution_completed"
    code: str
    block_index: int
    stdout: str
    stderr: str
    final_answer: str | None = None


@dataclass(frozen=True, kw_only=True)
class HelperQuestionGenerated(RLMEvent):
    event_type: ClassVar[str] = "helper_question_generated"
    question: str
    subcall_id: str
    subcall_kind: str
    batch_index: int | None = None


@dataclass(frozen=True, kw_only=True)
class PlainSubcallStarted(RLMEvent):
    event_type: ClassVar[str] = "plain_subcall_started"
    subcall_id: str
    prompt: str
    batch_index: int | None = None


@dataclass(frozen=True, kw_only=True)
class RecursiveSubcallStarted(RLMEvent):
    event_type: ClassVar[str] = "recursive_subcall_started"
    subcall_id: str
    child_invocation_id: str
    prompt: str
    batch_index: int | None = None


@dataclass(frozen=True, kw_only=True)
class SubcallCompleted(RLMEvent):
    event_type: ClassVar[str] = "subcall_completed"
    subcall_id: str
    response: str
    subcall_kind: str
    child_invocation_id: str | None = None
    prompt_token_ids: tuple[int, ...] = ()
    token_ids: tuple[int, ...] = ()
    token_offsets: tuple[tuple[int, int], ...] = ()
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class FinalAnswerSubmitted(RLMEvent):
    event_type: ClassVar[str] = "final_answer_submitted"
    answer: str


@dataclass(frozen=True, kw_only=True)
class InvocationCompleted(RLMEvent):
    event_type: ClassVar[str] = "invocation_completed"
    result: str
    execution_time: float


@dataclass(frozen=True, kw_only=True)
class InvocationFailed(RLMEvent):
    event_type: ClassVar[str] = "invocation_failed"
    error_type: str
    error: str


ExecutionEvent = (
    InvocationStarted
    | StudentGenerationStarted
    | StudentGenerationCompleted
    | CodeExecutionStarted
    | CodeExecutionCompleted
    | HelperQuestionGenerated
    | PlainSubcallStarted
    | RecursiveSubcallStarted
    | SubcallCompleted
    | FinalAnswerSubmitted
    | InvocationCompleted
    | InvocationFailed
)


_EVENT_TYPES: dict[str, type[RLMEvent]] = {
    event_type.event_type: event_type
    for event_type in (
        InvocationStarted,
        StudentGenerationStarted,
        StudentGenerationCompleted,
        CodeExecutionStarted,
        CodeExecutionCompleted,
        HelperQuestionGenerated,
        PlainSubcallStarted,
        RecursiveSubcallStarted,
        SubcallCompleted,
        FinalAnswerSubmitted,
        InvocationCompleted,
        InvocationFailed,
    )
}


def event_from_dict(payload: dict[str, Any]) -> ExecutionEvent:
    """Restore a validated event from its wire representation."""
    data = dict(payload)
    event_name = str(data.pop("event_type"))
    try:
        event_class = _EVENT_TYPES[event_name]
    except KeyError as exc:
        raise ValueError(f"unsupported RLM event type {event_name!r}") from exc
    data["producer"] = EventProducer(data["producer"])
    for name in ("prompt_token_ids", "token_ids", "token_offsets"):
        if name in data:
            data[name] = tuple(
                tuple(value) if isinstance(value, list) else value for value in data[name]
            )
    return event_class(**data)  # type: ignore[return-value]


__all__ = [
    "CodeExecutionCompleted",
    "CodeExecutionStarted",
    "EventProducer",
    "ExecutionEvent",
    "FinalAnswerSubmitted",
    "HelperQuestionGenerated",
    "InvocationCompleted",
    "InvocationFailed",
    "InvocationStarted",
    "PlainSubcallStarted",
    "RLMEvent",
    "RecursiveSubcallStarted",
    "StudentGenerationCompleted",
    "StudentGenerationStarted",
    "SubcallCompleted",
    "event_from_dict",
]
