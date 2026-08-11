"""Calculate self-distillation loss from ordinary and feedback-conditioned predictions."""

from __future__ import annotations

from rlm_train.objectives.sdpo.divergence import reverse_kl_topk_with_tail
from rlm_train.training.prepare_batch import LossResult, TrainingBatch


def calculate_loss(batch: TrainingBatch) -> LossResult:
    torch = __import__("torch")
    total = None
    active_total = 0
    for attempt in batch.attempts:
        attempt_id = attempt.rollout_id
        student_logits = batch.student_predictions.logits[attempt_id]
        if student_logits.ndim != 2:
            raise ValueError("SDPO student predictions must be [selected_tokens, vocabulary]")
        predictions = batch.feedback_predictions[attempt_id]
        selected = student_logits.shape[0]
        if selected != len(predictions.selected_positions) or not predictions.topk_token_ids:
            raise ValueError("student and feedback-conditioned predictions must align")
        device = student_logits.device
        student_logprobs = torch.log_softmax(student_logits.float(), dim=-1)
        topk_ids = torch.tensor(predictions.topk_token_ids, dtype=torch.long, device=device)
        feedback_topk = torch.tensor(predictions.topk_logprobs, dtype=torch.float32, device=device)
        feedback_tail = torch.tensor(
            predictions.tail_logprob_mass, dtype=torch.float32, device=device
        )
        student_topk = student_logprobs.gather(dim=-1, index=topk_ids)
        tail_values = student_logprobs.clone()
        tail_values.scatter_(dim=-1, index=topk_ids, value=float("-inf"))
        student_tail = torch.logsumexp(tail_values, dim=-1)
        mask = torch.ones(selected, dtype=torch.bool, device=device)
        loss = reverse_kl_topk_with_tail(
            student_topk,
            student_tail,
            feedback_topk,
            feedback_tail,
            mask,
        )
        total = loss * selected if total is None else total + loss * selected
        active_total += selected
    if active_total == 0:
        raise ValueError("SDPO computed no active tokens")
    return LossResult(
        loss=total / active_total,
        active_token_count=active_total,
        diagnostics={"sdpo/attempts": float(len(batch.attempts))},
    )


__all__ = ["calculate_loss"]
