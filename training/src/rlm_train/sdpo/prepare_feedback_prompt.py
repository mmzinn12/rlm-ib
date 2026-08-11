"""Construct feedback-conditioned messages and apply the real chat template."""

from __future__ import annotations

from rlm_train.attempts.attempt_records import AnnotatedAttempt
from rlm_train.feedback.feedback_records import ScopedAssessment


def generation_edge_ids(attempt: AnnotatedAttempt, generation_id: str) -> frozenset[str]:
    generation = next(
        item for item in attempt.annotations.generations if item.generation_id == generation_id
    )
    active_generation_by_invocation: dict[str, str] = {}
    lifecycle_seen = False
    edge_ids: set[str] = set()
    for event in attempt.execution.events:
        event_type = str(event.get("event_type") or "")
        invocation_id = str(event.get("invocation_id") or "")
        if event_type in {"student_generation_started", "student_generation_completed"}:
            if invocation_id == generation.node_id:
                lifecycle_seen = True
            active_generation_by_invocation[invocation_id] = str(event.get("generation_id") or "")
        elif (
            event_type == "helper_question_generated"
            and active_generation_by_invocation.get(invocation_id) == generation_id
        ):
            edge_ids.add(str(event["subcall_id"]))
    if lifecycle_seen:
        return frozenset(edge_ids)
    return frozenset(
        edge.edge_id for edge in attempt.execution.edges if edge.parent_id == generation.node_id
    )


def relevant_feedback_text(
    attempt: AnnotatedAttempt,
    generation_id: str,
    assessments: tuple[ScopedAssessment, ...],
) -> str:
    edge_ids = generation_edge_ids(attempt, generation_id)
    lines: list[str] = []
    for assessment in assessments:
        if not (set(assessment.focal_edge_ids) & edge_ids):
            continue
        rubric = (assessment.content or {}).get("rubric") or {}
        guidance = str(rubric.get("improved_question_guidance") or "").strip()
        missing = str(rubric.get("what_was_missing") or "").strip()
        if guidance:
            lines.append(f"- Improved helper question: {guidance}")
        if missing:
            lines.append(f"- Missing from the previous attempt: {missing}")
    return "\n".join(lines)


def original_generation_prompt(attempt: AnnotatedAttempt, generation_id: str) -> object:
    generation = next(
        item for item in attempt.annotations.generations if item.generation_id == generation_id
    )
    node = next(item for item in attempt.execution.nodes if item.node_id == generation.node_id)
    return node.prompt


def prepare_feedback_messages(
    attempt: AnnotatedAttempt,
    generation_id: str,
    assessments: tuple[ScopedAssessment, ...],
    *,
    normalize_messages: object,
) -> list[dict[str, str]]:
    """Reconstruct the original message context, then append permitted feedback."""
    if not callable(normalize_messages):
        raise TypeError("normalize_messages must be the student's prompt formatter")
    messages = list(normalize_messages(original_generation_prompt(attempt, generation_id)))
    feedback_text = relevant_feedback_text(attempt, generation_id, assessments)
    if feedback_text:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Use this feedback about the previous attempt when producing the response.\n"
                    f"{feedback_text}"
                ),
            }
        )
    return messages


__all__ = [
    "generation_edge_ids",
    "original_generation_prompt",
    "prepare_feedback_messages",
    "relevant_feedback_text",
]
