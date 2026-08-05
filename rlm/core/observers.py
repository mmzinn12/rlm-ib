"""Observer contracts for non-invasive RLM execution instrumentation."""

from __future__ import annotations

import copy
import threading
import uuid
from collections.abc import Iterable
from typing import Protocol, TypeVar

from rlm.core.events import EventProducer, ExecutionEvent, RLMEvent


class RLMObserver(Protocol):
    """Consume immutable events without changing execution behavior."""

    def observe(self, event: ExecutionEvent) -> None: ...


class NoOpObserver:
    """Default observer used when execution is not being recorded."""

    def observe(self, event: ExecutionEvent) -> None:
        del event


class CompositeObserver:
    """Fan each event out to multiple independent observers."""

    def __init__(self, observers: Iterable[RLMObserver]):
        self.observers = tuple(observers)

    def observe(self, event: ExecutionEvent) -> None:
        for observer in self.observers:
            observer.observe(event)


class RecordingObserver:
    """Thread-safe in-memory observer intended for tests and adapters."""

    def __init__(self) -> None:
        self._events: list[ExecutionEvent] = []
        self._lock = threading.Lock()

    def observe(self, event: ExecutionEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[ExecutionEvent, ...]:
        with self._lock:
            return tuple(sorted(self._events, key=lambda item: item.sequence_number))


EventT = TypeVar("EventT", bound=RLMEvent)


class EventEmitter:
    """Assign rollout-wide sequence numbers and stable event/invocation IDs."""

    def __init__(self, observer: RLMObserver, rollout_id: str | None = None):
        self.observer = observer
        self.rollout_id = rollout_id or str(uuid.uuid4())
        self._sequence = 0
        self._invocation_sequence = 0
        self._generation_sequence = 0
        self._subcall_sequence = 0
        self._lock = threading.Lock()

    def new_invocation_id(self) -> str:
        with self._lock:
            value = self._invocation_sequence
            self._invocation_sequence += 1
        return f"{self.rollout_id}/node/{value:06d}"

    def new_generation_id(self) -> str:
        with self._lock:
            value = self._generation_sequence
            self._generation_sequence += 1
        return f"{self.rollout_id}/generation/{value:06d}"

    def new_subcall_id(self) -> str:
        with self._lock:
            value = self._subcall_sequence
            self._subcall_sequence += 1
        return f"{self.rollout_id}/subcall/{value:06d}"

    def emit(
        self,
        event_class: type[EventT],
        *,
        invocation_id: str,
        parent_invocation_id: str | None,
        producer: EventProducer,
        policy_owner: str | None = None,
        source_model: str | None = None,
        **payload: object,
    ) -> EventT:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        event = event_class(
            rollout_id=self.rollout_id,
            event_id=f"{self.rollout_id}/event/{sequence:08d}",
            invocation_id=invocation_id,
            parent_invocation_id=parent_invocation_id,
            sequence_number=sequence,
            producer=producer,
            policy_owner=policy_owner,
            source_model=source_model,
            **copy.deepcopy(payload),
        )
        self.observer.observe(event)
        return event


__all__ = [
    "CompositeObserver",
    "EventEmitter",
    "NoOpObserver",
    "RLMObserver",
    "RecordingObserver",
]
