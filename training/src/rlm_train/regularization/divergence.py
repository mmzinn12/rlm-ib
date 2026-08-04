"""Compute detached per-token Jensen-Shannon drift over aligned model logits.

Purpose:
    Quantify functional student/reference drift for token sampling and optional Gram-pair
    weighting without introducing a differentiable auxiliary logit loss.
Implementation:
    Student and reference logits are detached, converted either to full distributions or
    a shared reference-top-k-plus-tail support, and compared with symmetric JS divergence.
Inputs:
    Causally aligned student/reference logit tensors, a vocabulary-support policy, and
    an optional top-k width.
Outputs:
    Detached per-position JS tensors or validated coarsened probability transports.
Example:
    ``token_js = per_token_js_divergence(student_logits, anchor_logits, top_k=100)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ReferenceLogitSource(Protocol):
    """Define an aligned, feedback-free reference-logit source.

    Implementations must teacher-force exactly the supplied token IDs and masks. Judge
    feedback, privileged evidence, or independently generated continuations are outside
    this interface.
    """

    @property
    def identity(self) -> Any:
        """Return versioned reference identity for metrics and caches."""
        ...

    def logits_for(self, inputs: Any) -> Any:
        """Teacher-force aligned inputs and return next-token logits.

        Args:
            inputs: Student-aligned sequence transport accepted by the implementation.

        Returns:
            Logits with the same sequence positions and vocabulary as the student.
        """
        ...


@dataclass(frozen=True)
class CoarsenedDistributions:
    """Hold distributions on shared reference top-k IDs plus a tail bucket.

    Attributes:
        token_ids: Reference-selected token IDs with shape ``[..., tokens, top_k]``.
        student_probabilities: Student mass over selected IDs and one final tail bucket.
        reference_probabilities: Reference mass over the identical coarsened support.

    Example:
        ``coarsened = coarsen_logits_to_reference_topk(student, anchor, top_k=100)``
    """

    token_ids: Any
    student_probabilities: Any
    reference_probabilities: Any


def coarsen_logits_to_reference_topk(
    student_logits: Any, reference_logits: Any, *, top_k: int
) -> CoarsenedDistributions:
    """Coarsen aligned logits using reference top-k IDs and one tail bucket.

    Args:
        student_logits: Student logits shaped ``[..., tokens, vocabulary]``.
        reference_logits: Shape-aligned feedback-free reference logits.
        top_k: Positive number of explicit reference-selected token IDs.

    Returns:
        Detached FP32 categorical probabilities on a shared ``top_k + 1`` support.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If logits are misaligned/non-finite, ``top_k`` is invalid, or the
            resulting probability mass fails validation.
    """
    torch = _torch()
    _validate_aligned_logits(student_logits, reference_logits)
    vocabulary_size = student_logits.shape[-1]
    if top_k <= 0 or top_k > vocabulary_size:
        raise ValueError("top_k must be positive and no larger than the vocabulary")

    student_log_probs = torch.log_softmax(student_logits.detach().float(), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.detach().float(), dim=-1)
    _, token_ids = torch.topk(reference_log_probs, k=top_k, dim=-1)
    student_top = student_log_probs.gather(-1, token_ids).exp()
    reference_top = reference_log_probs.gather(-1, token_ids).exp()
    student_tail = (1.0 - student_top.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    reference_tail = (1.0 - reference_top.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    student = torch.cat((student_top, student_tail), dim=-1)
    reference = torch.cat((reference_top, reference_tail), dim=-1)
    validate_probability_mass(student)
    validate_probability_mass(reference)
    return CoarsenedDistributions(token_ids, student, reference)


def per_token_js_divergence(
    student_logits: Any,
    reference_logits: Any,
    *,
    vocabulary_support: str = "reference_topk_tail",
    top_k: int = 100,
) -> Any:
    """Return detached JS divergence at each causally aligned hidden-state position.

    Position ``t`` compares ``logits[t]`` from both models; those logits predict the
    next token and are associated with hidden state ``H[t]`` for Gram selection. Both
    inputs are detached here so JS controls sampling/weights but is never a hidden
    differentiable logit loss.

    Args:
        student_logits: Student logits shaped ``[..., tokens, vocabulary]``.
        reference_logits: Reference logits with identical shape and causal alignment.
        vocabulary_support: ``reference_topk_tail`` or ``full``.
        top_k: Explicit support width used by top-k-plus-tail mode.

    Returns:
        Detached FP32 JS values shaped like the logit tensors without vocabulary.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If shapes, values, support mode, or probability mass are invalid.
    """
    torch = _torch()
    functional = __import__("torch.nn.functional", fromlist=["kl_div"])
    _validate_aligned_logits(student_logits, reference_logits)
    if vocabulary_support == "reference_topk_tail":
        coarsened = coarsen_logits_to_reference_topk(student_logits, reference_logits, top_k=top_k)
        student = coarsened.student_probabilities
        reference = coarsened.reference_probabilities
    elif vocabulary_support == "full":
        student = torch.softmax(student_logits.detach().float(), dim=-1)
        reference = torch.softmax(reference_logits.detach().float(), dim=-1)
    else:
        raise ValueError("unsupported JS vocabulary support")
    mixture = 0.5 * (student + reference)
    tiny = torch.finfo(mixture.dtype).tiny
    mixture_log = mixture.clamp_min(tiny).log()
    student_kl = functional.kl_div(
        mixture_log,
        student,
        reduction="none",
    ).sum(dim=-1)
    reference_kl = functional.kl_div(
        mixture_log,
        reference,
        reduction="none",
    )
    result = 0.5 * (student_kl + reference_kl.sum(dim=-1))
    if not torch.isfinite(result).all() or (result < -1e-7).any():
        raise ValueError("JS divergence must be finite and non-negative")
    return result.clamp_min(0.0).detach()


def validate_probability_mass(probabilities: Any, *, tolerance: float = 1e-5) -> None:
    """Validate non-negative finite categorical rows with unit mass.

    Args:
        probabilities: Tensor whose final dimension is categorical probability mass.
        tolerance: Absolute and relative tolerance for unit-mass comparison.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If the categorical dimension is empty, values are invalid, or rows
            do not sum to one.
    """
    torch = _torch()
    if probabilities.ndim == 0 or probabilities.shape[-1] == 0:
        raise ValueError("probabilities require a non-empty categorical dimension")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    expected = torch.ones_like(probabilities.sum(dim=-1))
    if not torch.allclose(probabilities.sum(dim=-1), expected, atol=tolerance, rtol=tolerance):
        raise ValueError("probability mass must sum to one")


def _validate_aligned_logits(student_logits: Any, reference_logits: Any) -> None:
    """Require identical sequence/vocabulary geometry and finite values."""
    torch = _torch()
    if student_logits.shape != reference_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and reference logits must have identical aligned shapes")
    if student_logits.shape[-1] <= 0:
        raise ValueError("logits require a non-empty vocabulary")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(reference_logits).all():
        raise ValueError("logits must be finite")


def _torch() -> Any:
    """Import PyTorch only for tensor execution, keeping configuration import-light."""
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Gram regularization tensors") from exc


__all__ = [
    "CoarsenedDistributions",
    "ReferenceLogitSource",
    "coarsen_logits_to_reference_topk",
    "per_token_js_divergence",
    "validate_probability_mass",
]
