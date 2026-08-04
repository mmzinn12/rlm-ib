"""Compute reverse KL over teacher top-k tokens and an explicit tail bucket.

Purpose:
    Distill a feedback-conditioned teacher into the student using the agreed
    ``D_KL(student || detached_teacher)`` direction.
Implementation:
    Both distributions are coarsened to the teacher-selected top-k tokens plus one
    bucket for the remaining vocabulary. PyTorch's KL primitive operates in log space,
    detaches teacher values, and normalizes the result by active masked tokens.
Inputs:
    Student and teacher log-probabilities for top-k tokens and tail mass, plus a mask.
Outputs:
    A differentiable PyTorch scalar loss.
Example:
    ``loss = reverse_kl_topk_with_tail(student_topk, student_tail, teacher_topk, teacher_tail, mask)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm.core.trajectory import DecisionKind

from rlm_train.sdpo.config import ComponentWeights
from rlm_train.sdpo.teacher import TopKTeacherTarget


@dataclass(frozen=True)
class StudentTopKDistribution:
    """Hold differentiable student top-k and aggregate-tail log-probabilities."""

    logprobs: Any
    tail_logprobs: Any


@dataclass(frozen=True)
class WeightedSDPOLoss:
    """Return the weighted objective and its normalized component diagnostics."""

    total: Any
    component_losses: dict[DecisionKind, Any]
    active_token_counts: dict[DecisionKind, int]


def reverse_kl_topk_with_tail(
    student_logprobs: Any,
    student_tail_logprobs: Any,
    teacher_logprobs: Any,
    teacher_tail_logprobs: Any,
    mask: Any,
) -> Any:
    """Compute masked reverse KL as a differentiable PyTorch scalar.

    Args:
        student_logprobs: Tensor shaped ``[..., tokens, top_k]``.
        student_tail_logprobs: Tensor shaped ``[..., tokens]``.
        teacher_logprobs: Teacher tensor matching ``student_logprobs``.
        teacher_tail_logprobs: Teacher tensor matching the student tail tensor.
        mask: Boolean or numeric tensor shaped ``[..., tokens]``.

    Returns:
        Reverse KL summed over active positions and divided by active-token count.
        Gradients flow only through student tensors.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If tensor shapes disagree or the mask activates no tokens.

    Example:
        ``loss = reverse_kl_topk_with_tail(s_topk, s_tail, t_topk, t_tail, mask)``
    """
    try:
        functional = __import__("torch.nn.functional", fromlist=["kl_div"])
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor SDPO loss computation") from exc

    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError("student and teacher top-k tensors must have identical shapes")
    if student_tail_logprobs.shape != teacher_tail_logprobs.shape:
        raise ValueError("student and teacher tail tensors must have identical shapes")
    if student_logprobs.shape[:-1] != student_tail_logprobs.shape:
        raise ValueError("top-k and tail tensors must align on token positions")
    if mask.shape != student_tail_logprobs.shape:
        raise ValueError("mask must align with token positions")

    detached_teacher = teacher_logprobs.detach()
    detached_teacher_tail = teacher_tail_logprobs.detach()
    token_kl = functional.kl_div(
        detached_teacher,
        student_logprobs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    token_kl = token_kl + functional.kl_div(
        detached_teacher_tail,
        student_tail_logprobs,
        reduction="none",
        log_target=True,
    )
    active = mask.to(dtype=token_kl.dtype).sum()
    if active.item() == 0:
        raise ValueError("reverse KL received no active tokens")
    return (token_kl * mask.to(dtype=token_kl.dtype)).sum() / active


def gather_student_topk_with_tail(
    student_logits: Any,
    teacher_target: TopKTeacherTarget,
) -> StudentTopKDistribution:
    """Gather differentiable student mass at teacher IDs and aggregate the tail.

    Args:
        student_logits: Tensor shaped ``[tokens, vocabulary]``.
        teacher_target: Teacher IDs defining the coarsened vocabulary per position.

    Returns:
        Student log-probabilities for those IDs and for every remaining token.

    Raises:
        TypeError: If logits are not a PyTorch tensor.
        ValueError: If shapes, IDs, or vocabulary coverage are invalid.
    """
    try:
        torch = __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for student probability gathering") from exc
    if not isinstance(student_logits, torch.Tensor):
        raise TypeError("student logits must be a PyTorch tensor")
    if student_logits.ndim != 2:
        raise ValueError("student logits must have shape [tokens, vocabulary]")
    token_count, vocabulary_size = student_logits.shape
    if token_count != len(teacher_target.token_ids):
        raise ValueError("student positions must match the teacher target")
    top_k = len(teacher_target.token_ids[0])
    if top_k >= vocabulary_size:
        raise ValueError("teacher top-k must leave at least one student tail token")
    if not torch.isfinite(student_logits).all().item():
        raise ValueError("student logits must be finite")
    token_ids = torch.as_tensor(
        teacher_target.token_ids,
        device=student_logits.device,
        dtype=torch.long,
    )
    if token_ids.max().item() >= vocabulary_size:
        raise ValueError("teacher target contains a token outside the student vocabulary")
    logprobs = torch.log_softmax(student_logits, dim=-1)
    selected = logprobs.gather(dim=-1, index=token_ids)
    tail_values = logprobs.scatter(
        dim=-1,
        index=token_ids,
        value=-torch.inf,
    )
    tail = torch.logsumexp(tail_values, dim=-1)
    return StudentTopKDistribution(logprobs=selected, tail_logprobs=tail)


def teacher_target_tensors(
    target: TopKTeacherTarget,
    *,
    reference: Any,
) -> tuple[Any, Any]:
    """Convert a validated target to detached tensors matching a student reference."""
    try:
        torch = __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for teacher tensor conversion") from exc
    if not isinstance(reference, torch.Tensor):
        raise TypeError("reference must be a PyTorch tensor")
    topk = torch.as_tensor(
        target.logprobs,
        device=reference.device,
        dtype=reference.dtype,
    ).detach()
    tail = torch.as_tensor(
        target.tail_logprobs,
        device=reference.device,
        dtype=reference.dtype,
    ).detach()
    return topk, tail


def weighted_component_reverse_kl(
    student_logits: Any,
    teacher_target: TopKTeacherTarget,
    component_masks: dict[DecisionKind, list[bool] | Any],
    component_weights: ComponentWeights,
) -> WeightedSDPOLoss:
    """Normalize each active component by tokens and return their weighted sum."""
    try:
        torch = __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for component SDPO computation") from exc
    student = gather_student_topk_with_tail(student_logits, teacher_target)
    teacher_topk, teacher_tail = teacher_target_tensors(
        teacher_target,
        reference=student.logprobs,
    )
    weights = {
        DecisionKind.ROUTE: component_weights.route,
        DecisionKind.CALL: component_weights.call,
        DecisionKind.NODE: component_weights.node,
        DecisionKind.AGGREGATION: component_weights.aggregation,
        DecisionKind.FINAL: component_weights.final,
        DecisionKind.MISSING_CALL: component_weights.missing_call,
    }
    component_losses: dict[DecisionKind, Any] = {}
    active_token_counts: dict[DecisionKind, int] = {}
    weighted_losses: list[Any] = []
    token_count = student.tail_logprobs.shape[0]
    masks: dict[DecisionKind, Any] = {}
    for kind, mask_value in component_masks.items():
        if kind not in weights:
            raise ValueError(f"unsupported SDPO component {kind!r}")
        mask = torch.as_tensor(
            mask_value,
            device=student_logits.device,
            dtype=torch.bool,
        )
        if mask.shape != (token_count,):
            raise ValueError(f"{kind.value} mask must align with continuation tokens")
        masks[kind] = mask
    if masks and torch.stack(list(masks.values())).sum(dim=0).gt(1).any().item():
        raise ValueError("SDPO component masks must be exclusive")
    for kind, mask in masks.items():
        active_count = int(mask.sum().item())
        if active_count == 0 or weights[kind] == 0:
            continue
        loss = reverse_kl_topk_with_tail(
            student.logprobs,
            student.tail_logprobs,
            teacher_topk,
            teacher_tail,
            mask,
        )
        component_losses[kind] = loss
        active_token_counts[kind] = active_count
        weighted_losses.append(loss * weights[kind])
    if not weighted_losses:
        raise ValueError("weighted SDPO objective has no active, non-zero component")
    return WeightedSDPOLoss(
        total=torch.stack(weighted_losses).sum(),
        component_losses=component_losses,
        active_token_counts=active_token_counts,
    )


__all__ = [
    "StudentTopKDistribution",
    "WeightedSDPOLoss",
    "gather_student_topk_with_tail",
    "reverse_kl_topk_with_tail",
    "teacher_target_tensors",
    "weighted_component_reverse_kl",
]
