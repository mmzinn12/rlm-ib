"""SDPO divergence implementations."""

from __future__ import annotations

from typing import Any


def reverse_kl_topk_with_tail(
    student_logprobs: Any,
    student_tail_logprobs: Any,
    teacher_logprobs: Any,
    teacher_tail_logprobs: Any,
    mask: Any,
) -> Any:
    """Compute masked ``D_KL(student || detached_teacher)`` on top-k plus tail."""
    try:
        functional = __import__("torch.nn.functional", fromlist=["kl_div"])
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SDPO") from exc
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError("student and teacher top-k tensors must have identical shapes")
    if student_tail_logprobs.shape != teacher_tail_logprobs.shape:
        raise ValueError("student and teacher tail tensors must have identical shapes")
    if student_logprobs.shape[:-1] != student_tail_logprobs.shape:
        raise ValueError("top-k and tail tensors must align on token positions")
    if mask.shape != student_tail_logprobs.shape:
        raise ValueError("mask must align with token positions")
    token_kl = functional.kl_div(
        teacher_logprobs.detach(),
        student_logprobs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    token_kl = token_kl + functional.kl_div(
        teacher_tail_logprobs.detach(),
        student_tail_logprobs,
        reduction="none",
        log_target=True,
    )
    weights = mask.to(dtype=token_kl.dtype)
    active = weights.sum()
    if active.item() == 0:
        raise ValueError("reverse KL received no active tokens")
    return (token_kl * weights).sum() / active


__all__ = ["reverse_kl_topk_with_tail"]
