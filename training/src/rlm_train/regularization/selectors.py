"""Resolve fixed Gram layers and construct valid token-position masks.

Purpose:
    Freeze model-relative layer choices at initialization and decouple token-scope
    selection from trajectory segmentation and trainer internals.
Implementation:
    Relative depths map deterministically to zero-based block indices, collisions are
    resolved with stable weighting, and boolean helpers combine attention, completion,
    decision, special-position, and alignment masks.
Inputs:
    Layer-selection configuration, transformer block count, attention masks, completion
    boundaries, and optional decision/special/alignment masks.
Outputs:
    ``ResolvedLayerSelection`` metadata and boolean token masks.
Example:
    ``layers = resolve_layer_selection(config.layers, block_count=32)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rlm_train.regularization.config import GramLayerSelectionConfig


@dataclass(frozen=True)
class ResolvedLayerSelection:
    """Freeze requested layers, concrete indices, and aggregation weights.

    Attributes:
        requested_indices: Original explicit indices, when that mode was selected.
        requested_relative_depths: Original relative depths, when selected.
        indices: Unique resolved zero-based transformer-block indices.
        weights: Non-negative weights aligned with ``indices``.
        block_count: Transformer block count used during resolution.
    """

    requested_indices: tuple[int, ...] | None
    requested_relative_depths: tuple[float, ...] | None
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    block_count: int


def resolve_layer_selection(
    config: GramLayerSelectionConfig, *, block_count: int
) -> ResolvedLayerSelection:
    """Resolve relative depths once using ``ceil(depth * N) - 1``.

    Relative depths that collide on a shallow model are deduplicated. Explicit weights
    for collided depths are summed; implicit weights remain one per resolved layer.

    Args:
        config: Validated explicit-index or relative-depth policy.
        block_count: Positive number of transformer blocks in the instantiated model.

    Returns:
        Frozen requested and resolved layer metadata.

    Raises:
        ValueError: If ``block_count`` is non-positive, an explicit index is out of
            range, or resolved indices/weights violate uniqueness and alignment.

    Example:
        ``resolved = resolve_layer_selection(config, block_count=32)``
    """
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    requested_weights = config.layer_weights
    if config.indices is not None:
        if any(index >= block_count for index in config.indices):
            raise ValueError("explicit layer index exceeds the transformer block count")
        indices = config.indices
        weights = requested_weights or tuple(1.0 for _ in indices)
    else:
        depths = config.relative_depths or ()
        raw_indices = tuple(
            min(block_count - 1, max(0, math.ceil(depth * block_count) - 1)) for depth in depths
        )
        index_order: list[int] = []
        weight_by_index: dict[int, float] = {}
        for position, index in enumerate(raw_indices):
            if index not in weight_by_index:
                index_order.append(index)
                weight_by_index[index] = 0.0 if requested_weights is not None else 1.0
            if requested_weights is not None:
                weight_by_index[index] += requested_weights[position]
        indices = tuple(index_order)
        weights = tuple(weight_by_index[index] for index in indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("resolved layer indices must be non-empty and unique")
    if len(weights) != len(indices) or any(weight < 0.0 for weight in weights):
        raise ValueError("resolved layer weights must align and be non-negative")
    return ResolvedLayerSelection(
        requested_indices=config.indices,
        requested_relative_depths=config.relative_depths,
        indices=indices,
        weights=weights,
        block_count=block_count,
    )


def build_completion_mask(
    attention_mask: Any,
    completion_start_positions: int | list[int] | Any,
    *,
    special_position_mask: Any | None = None,
) -> Any:
    """Include attended positions at or after each completion boundary.

    Args:
        attention_mask: One- or two-dimensional attended-position mask.
        completion_start_positions: Scalar or one start position per batch row.
        special_position_mask: Optional aligned mask whose true positions are excluded.

    Returns:
        Boolean mask with the same unbatched/batched shape as ``attention_mask``.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If masks/boundaries are misaligned or a boundary is out of range.
    """
    torch = _torch()
    attention = _boolean_mask(attention_mask, name="attention_mask")
    squeeze = attention.ndim == 1
    batched = attention.unsqueeze(0) if squeeze else attention
    if batched.ndim != 2:
        raise ValueError("attention_mask must be one- or two-dimensional")
    starts = torch.as_tensor(completion_start_positions, device=batched.device)
    if starts.ndim == 0:
        starts = starts.repeat(batched.shape[0])
    if starts.shape != (batched.shape[0],):
        raise ValueError("completion start positions must align with the batch")
    if (starts < 0).any() or (starts > batched.shape[1]).any():
        raise ValueError("completion start positions must lie within the sequence")
    positions = torch.arange(batched.shape[1], device=batched.device).unsqueeze(0)
    result = batched & (positions >= starts.unsqueeze(1))
    if special_position_mask is not None:
        special = _boolean_mask_like(special_position_mask, attention, name="special_position_mask")
        special = special.unsqueeze(0) if squeeze else special
        result &= ~special
    return result.squeeze(0) if squeeze else result


def build_valid_token_mask(
    attention_mask: Any,
    *,
    token_scope: str = "completion",
    completion_mask: Any | None = None,
    decision_mask: Any | None = None,
    special_position_mask: Any | None = None,
    alignment_mask: Any | None = None,
) -> Any:
    """Build a mask for aligned, non-padding, non-special positions.

    Args:
        attention_mask: Base attended-position mask.
        token_scope: ``completion``, ``all_valid``, or ``decision``.
        completion_mask: Required aligned mask for completion scope.
        decision_mask: Required aligned mask for decision scope.
        special_position_mask: Optional true-for-special exclusion mask.
        alignment_mask: Optional mask excluding positions without aligned model outputs.

    Returns:
        Boolean tensor aligned with ``attention_mask``.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If scope is unsupported, required scope masks are absent, or any
            supplied mask shape differs from the attention mask.
    """
    valid = _boolean_mask(attention_mask, name="attention_mask").clone()
    if token_scope == "completion":
        if completion_mask is None:
            raise ValueError("completion token scope requires completion_mask")
        valid &= _boolean_mask_like(completion_mask, valid, name="completion_mask")
    elif token_scope == "decision":
        if decision_mask is None:
            raise ValueError("decision token scope requires decision_mask")
        valid &= _boolean_mask_like(decision_mask, valid, name="decision_mask")
    elif token_scope != "all_valid":
        raise ValueError("unsupported token scope")
    if special_position_mask is not None:
        valid &= ~_boolean_mask_like(special_position_mask, valid, name="special_position_mask")
    if alignment_mask is not None:
        valid &= _boolean_mask_like(alignment_mask, valid, name="alignment_mask")
    return valid


def _boolean_mask(value: Any, *, name: str) -> Any:
    """Convert tensor-like input to a boolean tensor and reject scalar masks."""
    torch = _torch()
    result = torch.as_tensor(value, dtype=torch.bool)
    if result.ndim == 0:
        raise ValueError(f"{name} must include a sequence dimension")
    return result


def _boolean_mask_like(value: Any, reference: Any, *, name: str) -> Any:
    """Convert and require exact mask alignment."""
    torch = _torch()
    result = torch.as_tensor(value, dtype=torch.bool, device=reference.device)
    if result.shape != reference.shape:
        raise ValueError(f"{name} must align with attention_mask")
    return result


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Gram regularization tensors") from exc


__all__ = [
    "ResolvedLayerSelection",
    "build_completion_mask",
    "build_valid_token_mask",
    "resolve_layer_selection",
]
