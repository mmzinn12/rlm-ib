"""Core event and observer contracts remain trainer-independent."""

from rlm.core.events import EventProducer, InvocationStarted, event_from_dict
from rlm.core.observers import EventEmitter, RecordingObserver


def test_event_roundtrip_and_rollout_sequence():
    observer = RecordingObserver()
    emitter = EventEmitter(observer, rollout_id="rollout")
    node_id = emitter.new_invocation_id()
    event = emitter.emit(
        InvocationStarted,
        invocation_id=node_id,
        parent_invocation_id=None,
        producer=EventProducer.RUNTIME,
        prompt="task",
        depth=0,
        node_role="root",
    )

    restored = event_from_dict(event.to_dict())

    assert restored == event
    assert observer.events == (event,)
    assert event.event_id == "rollout/event/00000000"
