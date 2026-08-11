"""Score exact selected continuations under feedback-conditioned prompts."""

from __future__ import annotations

import hashlib
import json

from rlm_train.attempts.attempt_records import AnnotatedAttempt
from rlm_train.feedback.feedback_records import FeedbackBundle, FeedbackVisibility
from rlm_train.generation.generated_text import GeneratedText
from rlm_train.objectives.sdpo.target_support import extract_topk_teacher_target
from rlm_train.sdpo.feedback_predictions import FeedbackPredictions
from rlm_train.sdpo.prepare_feedback_prompt import prepare_feedback_messages
from rlm_train.settings.token_selection import TokenScope
from rlm_train.student.student import TrainableStudent
from rlm_train.token_selection.selection import TokenSelection


def reconstruct_generated_text(
    attempt: AnnotatedAttempt,
    generation_id: str,
    student: TrainableStudent,
    *,
    prompt_token_ids: tuple[int, ...] | None = None,
) -> GeneratedText:
    generation = next(
        item for item in attempt.annotations.generations if item.generation_id == generation_id
    )
    return GeneratedText(
        text=generation.text,
        prompt_token_ids=prompt_token_ids or generation.prompt_token_ids,
        token_ids=generation.token_ids,
        token_offsets=generation.token_offsets,
        student=student.model_info,
        tokenizer=student.tokenizer_info,
    )


def score_with_feedback(
    *,
    student: TrainableStudent,
    attempts: tuple[AnnotatedAttempt, ...],
    feedback: FeedbackBundle,
    selections: dict[str, TokenSelection],
    included_text: TokenScope,
    top_k: int,
    version: int = 0,
) -> dict[str, FeedbackPredictions]:
    """Apply the chat template to full conditioned messages and score under no-grad."""
    if top_k <= 0:
        raise ValueError("SDPO top_k must be positive")
    formatter = getattr(getattr(student, "generator", None), "formatter", None)
    if formatter is None:
        raise TypeError("student must expose its generation PromptFormatter")
    settings_fingerprint = hashlib.sha256(
        json.dumps({"top_k": top_k, "version": version}, sort_keys=True).encode()
    ).hexdigest()
    results: dict[str, FeedbackPredictions] = {}
    for attempt in attempts:
        selection = selections[attempt.rollout_id]
        if selection.active_token_count == 0:
            raise ValueError("feedback scoring requires a non-empty token selection")
        permitted_assessments = tuple(
            assessment
            for assessment in feedback.local_assessments
            if (not assessment.allowed_objectives or "sdpo" in assessment.allowed_objectives)
            and (
                not assessment.allowed_token_scopes
                or included_text.value in assessment.allowed_token_scopes
            )
        )
        if any(
            assessment.visibility is FeedbackVisibility.PRIVILEGED
            for assessment in permitted_assessments
        ):
            raise ValueError("privileged verifier feedback is not permitted in student prompts")
        token_ids: list[int] = []
        positions: list[int] = []
        generation_ids: list[str] = []
        topk_ids: list[tuple[int, ...]] = []
        topk_logprobs: list[tuple[float, ...]] = []
        tail_logprobs: list[float] = []
        for selected_generation in selection.generations:
            messages = prepare_feedback_messages(
                attempt,
                selected_generation.generation_id,
                permitted_assessments,
                normalize_messages=formatter.messages,
            )
            conditioned_prompt_ids = student.format_prompt(messages)
            generated_text = reconstruct_generated_text(
                attempt,
                selected_generation.generation_id,
                student,
                prompt_token_ids=conditioned_prompt_ids,
            )
            predictions = student.score_tokens(
                generated_text,
                with_gradients=False,
                return_logits=True,
                return_logprobs=False,
                positions=selected_generation.positions,
            )
            if predictions.logits is None or bool(predictions.logits.requires_grad):
                raise ValueError("feedback-conditioned logits must be present and detached")
            support = extract_topk_teacher_target(
                predictions.logits,
                top_k=top_k,
                teacher_version=version,
                tokenizer_fingerprint=student.tokenizer_info.resolved_fingerprint,
            )
            token_ids.extend(
                generated_text.token_ids[position] for position in selected_generation.positions
            )
            positions.extend(selected_generation.positions)
            generation_ids.extend(
                selected_generation.generation_id for _ in selected_generation.positions
            )
            topk_ids.extend(support.token_ids)
            topk_logprobs.extend(support.logprobs)
            tail_logprobs.extend(support.tail_logprobs)
        identity = {
            "attempt": attempt.rollout_id,
            "selection": [(item.generation_id, item.positions) for item in selection.generations],
            "student": student.model_info.resolved_fingerprint,
            "settings": settings_fingerprint,
        }
        results[attempt.rollout_id] = FeedbackPredictions(
            prediction_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            attempt_id=attempt.rollout_id,
            generation_id=selection.generations[0].generation_id,
            selected_generation_ids=tuple(generation_ids),
            selected_token_ids=tuple(token_ids),
            selected_positions=tuple(positions),
            topk_token_ids=tuple(topk_ids),
            topk_logprobs=tuple(topk_logprobs),
            tail_logprob_mass=tuple(tail_logprobs),
            student_fingerprint=student.model_info.resolved_fingerprint,
            tokenizer_fingerprint=student.tokenizer_info.resolved_fingerprint,
            feedback_assessment_ids=tuple(item.assessment_id for item in permitted_assessments),
            feedback_projection_ids=tuple(item.projection_id for item in feedback.projections),
            judge_view_fingerprints=tuple(
                fingerprint
                for item in feedback.projections
                for fingerprint in item.view_fingerprints
            ),
            feedback_visibility=tuple(item.visibility.value for item in feedback.projections),
            settings_fingerprint=settings_fingerprint,
        )
    return results


__all__ = ["reconstruct_generated_text", "score_with_feedback"]
