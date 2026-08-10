"""Concrete SDPO reverse-KL loss over the teacher's top-k support plus the tail bucket."""

from __future__ import annotations

from collections.abc import Callable

from rlm_train.objectives.protocol import ObjectiveBatch, ObjectiveResult
from rlm_train.objectives.sdpo.divergence import reverse_kl_topk_with_tail
from rlm_train.objectives.sdpo.objective import SDPOObjective
from rlm_train.spec.objectives import SDPOSpec


def build_sdpo_compute_loss(spec: SDPOSpec) -> Callable[[ObjectiveBatch], ObjectiveResult]:
    """Create the SDPO loss closure that distills the student toward the teacher distribution.

    The returned closure computes, per rollout and per selected token, the reverse KL from the
    student to the detached teacher over the teacher's top-k token support plus a single tail
    bucket (the mass outside the top-k). Student log-probs are gathered on the teacher's top-k ids
    and the student tail is the log-sum-exp of the remaining vocabulary. Per-rollout losses are
    summed weighted by their active-token count and normalized by the total, so every selected
    token contributes equally regardless of how rollouts are batched.

    Args:
        spec: SDPO configuration; retained for closure identity and future parameterization.

    Returns:
        A callable mapping an ``ObjectiveBatch`` to an ``ObjectiveResult`` (scalar loss plus the
        active-token count and diagnostics). The batch must provide, per rollout id, student
        ``policy_scores`` as ``[selected_tokens, vocabulary]`` logits and a teacher target whose
        top-k support aligns with the selected positions.
    """

    def compute_loss(batch: ObjectiveBatch) -> ObjectiveResult:
        """Compute the active-token-weighted reverse-KL loss for one objective batch.

        Args:
            batch: Prepared batch supplying, per rollout id, student ``policy_scores`` logits and
                the aligned teacher ``teacher_targets``.

        Returns:
            An ``ObjectiveResult`` with the scalar loss, the total active-token count, and the
            per-batch rollout diagnostic.

        Raises:
            ValueError: If student logits are not 2-D, student and teacher selections misalign, or
                no active tokens were selected across the batch.
        """
        torch = __import__("torch")
        total = None
        active_total = 0
        for rollout in batch.rollouts:
            rollout_id = rollout.rollout_id
            student_logits = batch.policy_scores[rollout_id]
            if student_logits.ndim != 2:
                raise ValueError("SDPO policy scores must be [selected_tokens, vocabulary] logits")
            target = batch.teacher_targets[rollout_id]
            selected = student_logits.shape[0]
            if selected != len(target.selected_positions) or not target.topk_token_ids:
                raise ValueError("student scores and teacher top-k targets must align")
            device = student_logits.device
            student_logprobs = torch.log_softmax(student_logits.float(), dim=-1)
            topk_ids = torch.tensor(target.topk_token_ids, dtype=torch.long, device=device)
            teacher_topk = torch.tensor(target.topk_logprobs, dtype=torch.float32, device=device)
            teacher_tail = torch.tensor(
                target.tail_logprob_mass, dtype=torch.float32, device=device
            )
            student_topk = student_logprobs.gather(dim=-1, index=topk_ids)
            tail_values = student_logprobs.clone()
            tail_values.scatter_(dim=-1, index=topk_ids, value=float("-inf"))
            student_tail = torch.logsumexp(tail_values, dim=-1)
            mask = torch.ones(selected, dtype=torch.bool, device=device)
            loss = reverse_kl_topk_with_tail(
                student_topk, student_tail, teacher_topk, teacher_tail, mask
            )
            total = loss * selected if total is None else total + loss * selected
            active_total += selected
        if active_total == 0:
            raise ValueError("SDPO computed no active tokens")
        return ObjectiveResult(
            loss=total / active_total,
            active_token_count=active_total,
            diagnostics={"sdpo/rollouts": float(len(batch.rollouts))},
        )

    return compute_loss


def build_sdpo_objective(spec: SDPOSpec) -> SDPOObjective:
    """Build an SDPO objective bound to its reverse-KL top-k+tail loss.

    Args:
        spec: SDPO configuration (token scope, feedback scope, ``top_k``, weight).

    Returns:
        An ``SDPOObjective`` whose ``compute`` distills the student toward the teacher targets.
    """
    return SDPOObjective(spec, build_sdpo_compute_loss(spec))


__all__ = ["build_sdpo_compute_loss", "build_sdpo_objective"]
