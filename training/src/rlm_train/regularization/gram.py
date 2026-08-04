"""Compute numerically stable weighted and unweighted Gram-geometry losses.

Purpose:
    Preserve relational geometry among selected student token representations relative
    to a detached anchor while allowing policy and SDPO objectives to adapt the model.
Implementation:
    Selected features are cast to FP32, optionally normalized, converted to token-token
    Gram matrices, compared with MSE, optionally JS-pair weighted, and layer aggregated.
Inputs:
    Student/anchor hidden states, sampled token positions and JS values, resolved layers,
    pair-weighting policy, normalization policy, and loss weights.
Outputs:
    Differentiable student-only Gram losses plus detached-compatible layer diagnostics.
Example:
    ``result = gram_matrix_loss(student_states, anchor_states, token_weights=token_js)``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlm_train.regularization.selectors import ResolvedLayerSelection


@dataclass(frozen=True)
class GramLayerLoss:
    """Expose optimized and unweighted diagnostic losses for one block.

    Attributes:
        weighted: Loss selected by ``pair_weighting`` and used for optimization.
        unweighted: Ordinary mean-squared Gram difference for diagnostics.
    """

    weighted: Any
    unweighted: Any


@dataclass(frozen=True)
class GramAnchorLossResult:
    """Carry the global auxiliary loss and per-layer diagnostics.

    Attributes:
        total_loss: Layer-aggregated loss after applying the global coefficient.
        unweighted_diagnostic_loss: Layer-aggregated ordinary Gram MSE.
        layer_losses: Optimized loss by resolved block index.
        unweighted_layer_losses: Ordinary diagnostic loss by block index.
        effective_layer_weights: Actual global coefficient applied to each block.
        sampled_positions: Shared sequence positions used at every selected layer.
    """

    total_loss: Any
    unweighted_diagnostic_loss: Any
    layer_losses: dict[int, Any]
    unweighted_layer_losses: dict[int, Any]
    effective_layer_weights: dict[int, float]
    sampled_positions: tuple[int, ...]


def gram_matrix_loss(
    student_hidden_states: Any,
    anchor_hidden_states: Any,
    *,
    token_weights: Any | None = None,
    pair_weighting: str = "js_product",
    normalize_hidden_states: bool = True,
) -> GramLayerLoss:
    """Compute one layer's Gram loss with student-only gradients.

    Args:
        student_hidden_states: Tensor shaped ``[sampled_tokens, hidden_width]``.
        anchor_hidden_states: Shape-aligned anchor tensor; detached internally.
        token_weights: Optional non-negative per-token JS-derived weights.
        pair_weighting: ``js_product`` or ``none``.
        normalize_hidden_states: Whether to L2-normalize features before dot products.

    Returns:
        Optimized weighted/unweighted loss and an ordinary diagnostic loss.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If hidden states or weights have invalid shapes/values, no token is
            present, or the pair-weighting policy is unsupported.

    Example:
        ``loss = gram_matrix_loss(student, anchor, token_weights=js).weighted``
    """
    torch = _torch()
    functional = __import__("torch.nn.functional", fromlist=["normalize"])
    if student_hidden_states.shape != anchor_hidden_states.shape:
        raise ValueError("student and anchor hidden states must have identical shapes")
    if student_hidden_states.ndim != 2 or student_hidden_states.shape[0] == 0:
        raise ValueError("hidden states must have shape [sampled_tokens, hidden_width]")
    if (
        not torch.isfinite(student_hidden_states).all()
        or not torch.isfinite(anchor_hidden_states).all()
    ):
        raise ValueError("hidden states must be finite")
    student = student_hidden_states.float()
    anchor = anchor_hidden_states.detach().float()
    if normalize_hidden_states:
        student = functional.normalize(student, p=2.0, dim=-1)
        anchor = functional.normalize(anchor, p=2.0, dim=-1)
    student_gram = student @ student.transpose(0, 1)
    anchor_gram = anchor @ anchor.transpose(0, 1)
    squared_error = (student_gram - anchor_gram).square()
    unweighted = squared_error.mean()
    if pair_weighting == "none":
        return GramLayerLoss(weighted=unweighted, unweighted=unweighted)
    if pair_weighting != "js_product":
        raise ValueError("unsupported Gram pair weighting")
    if token_weights is None:
        raise ValueError("JS-product pair weighting requires token_weights")
    weights = torch.as_tensor(
        token_weights, dtype=torch.float32, device=squared_error.device
    ).detach()
    if weights.shape != (student.shape[0],):
        raise ValueError("token weights must align with sampled hidden states")
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum().item() <= 0:
        raise ValueError("token weights must have finite, non-negative, positive mass")
    normalized = weights / weights.mean()
    pair_weights = normalized[:, None] * normalized[None, :]
    weighted = (pair_weights * squared_error).sum() / pair_weights.sum()
    return GramLayerLoss(weighted=weighted, unweighted=unweighted)


def multi_layer_gram_loss(
    student_hidden_states: Mapping[int, Any],
    anchor_hidden_states: Mapping[int, Any],
    *,
    selection: ResolvedLayerSelection,
    sampled_positions: Sequence[int],
    sampled_js_values: Sequence[float],
    loss_weight: float,
    pair_weighting: str = "js_product",
    normalize_hidden_states: bool = True,
    minimum_weight: float = 1e-8,
) -> GramAnchorLossResult:
    """Gather shared positions and aggregate normalized multi-layer Gram losses.

    Args:
        student_hidden_states: Student sequence states keyed by resolved block index.
        anchor_hidden_states: Shape-aligned anchor states keyed by the same indices.
        selection: Frozen resolved indices and non-negative layer weights.
        sampled_positions: Sequence positions shared by every layer.
        sampled_js_values: Detached JS values aligned with sampled positions.
        loss_weight: Non-negative global auxiliary coefficient.
        pair_weighting: ``js_product`` or ``none``.
        normalize_hidden_states: Whether to normalize feature vectors.
        minimum_weight: Positive epsilon added to each JS pair weight.

    Returns:
        A differentiable global loss, layer diagnostics, effective coefficients, and
        the positions used for all layers.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If layers, positions, weights, hidden-state shapes, or coefficients
            violate alignment and positivity invariants.

    Example:
        ``result = multi_layer_gram_loss(student_layers, anchor_layers, selection=layers, sampled_positions=positions, sampled_js_values=js, loss_weight=0.1)``
    """
    torch = _torch()
    if loss_weight < 0.0:
        raise ValueError("Gram loss_weight must be non-negative")
    if minimum_weight <= 0.0:
        raise ValueError("minimum_weight must be positive")
    positions = tuple(int(position) for position in sampled_positions)
    if not positions or any(position < 0 for position in positions):
        raise ValueError("sampled positions must be non-empty and non-negative")
    if len(sampled_js_values) != len(positions):
        raise ValueError("sampled JS values must align with sampled positions")
    if set(student_hidden_states) != set(selection.indices):
        raise ValueError("student hidden-state layers must match resolved layers")
    if set(anchor_hidden_states) != set(selection.indices):
        raise ValueError("anchor hidden-state layers must match resolved layers")
    if len(selection.weights) != len(selection.indices):
        raise ValueError("resolved layer weights must align with layer indices")
    if any(weight < 0.0 for weight in selection.weights):
        raise ValueError("layer weights must be non-negative")
    weight_sum = sum(selection.weights)
    if loss_weight > 0.0 and weight_sum <= 0.0:
        raise ValueError("an active Gram loss requires a positive layer weight")

    position_tensor_by_device: dict[Any, Any] = {}
    layer_losses: dict[int, Any] = {}
    diagnostics: dict[int, Any] = {}
    for layer_index in selection.indices:
        student_layer = student_hidden_states[layer_index]
        anchor_layer = anchor_hidden_states[layer_index]
        if student_layer.ndim != 2 or anchor_layer.ndim != 2:
            raise ValueError("captured hidden states must have shape [sequence, hidden_width]")
        if student_layer.shape != anchor_layer.shape:
            raise ValueError("captured student and anchor layer shapes must match")
        if max(positions) >= student_layer.shape[0]:
            raise ValueError("sampled position exceeds a captured sequence length")
        if student_layer.device not in position_tensor_by_device:
            position_tensor_by_device[student_layer.device] = torch.as_tensor(
                positions, dtype=torch.long, device=student_layer.device
            )
        indices = position_tensor_by_device[student_layer.device]
        student_selected = student_layer.index_select(0, indices)
        anchor_indices = indices.to(anchor_layer.device)
        anchor_selected = anchor_layer.index_select(0, anchor_indices)
        token_weights = [value + minimum_weight for value in sampled_js_values]
        result = gram_matrix_loss(
            student_selected,
            anchor_selected,
            token_weights=token_weights,
            pair_weighting=pair_weighting,
            normalize_hidden_states=normalize_hidden_states,
        )
        layer_losses[layer_index] = result.weighted
        diagnostics[layer_index] = result.unweighted

    first_loss = layer_losses[selection.indices[0]]
    if loss_weight == 0.0:
        total = first_loss * 0.0
    else:
        total = (
            loss_weight
            * sum(
                selection.weights[position] * layer_losses[layer_index]
                for position, layer_index in enumerate(selection.indices)
            )
            / weight_sum
        )
    diagnostic_weight_sum = weight_sum if weight_sum > 0.0 else float(len(selection.indices))
    diagnostic_weights = (
        selection.weights if weight_sum > 0.0 else tuple(1.0 for _ in selection.indices)
    )
    unweighted_diagnostic = (
        sum(
            diagnostic_weights[position] * diagnostics[layer_index]
            for position, layer_index in enumerate(selection.indices)
        )
        / diagnostic_weight_sum
    )
    return GramAnchorLossResult(
        total_loss=total,
        unweighted_diagnostic_loss=unweighted_diagnostic,
        layer_losses=layer_losses,
        unweighted_layer_losses=diagnostics,
        effective_layer_weights={
            layer_index: (
                loss_weight * selection.weights[position] / weight_sum if weight_sum > 0.0 else 0.0
            )
            for position, layer_index in enumerate(selection.indices)
        },
        sampled_positions=positions,
    )


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Gram regularization tensors") from exc


__all__ = [
    "GramAnchorLossResult",
    "GramLayerLoss",
    "gram_matrix_loss",
    "multi_layer_gram_loss",
]
