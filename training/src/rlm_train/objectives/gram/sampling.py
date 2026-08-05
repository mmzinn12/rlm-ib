"""Sample valid token positions reproducibly from detached JS scores.

Purpose:
    Bound quadratic Gram-matrix memory while prioritizing positions where student and
    reference behavior has drifted.
Implementation:
    A uniform/JS mixture is normalized over valid positions, a stable seed is derived
    from run metadata, and PyTorch multinomial sampling returns a replayable selection.
Inputs:
    One-dimensional JS scores and valid mask, sampling configuration, global step,
    stable sample ID, and distributed rank.
Outputs:
    ``TokenSampleSelection`` containing selected positions, scores, probabilities, and
    reproducibility metadata.
Example:
    ``selection = sample_token_positions(token_js, valid_mask, config, global_step=4, sample_id="row-7")``
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from rlm_train.objectives.gram.config import JSTokenSamplingConfig


@dataclass(frozen=True)
class TokenSampleSelection:
    """Record selected positions, valid-set probabilities, and replay metadata.

    Attributes:
        selected_positions: Sequence positions chosen for every Gram layer.
        selected_js_values: Raw JS values at the selected positions.
        valid_positions: All positions eligible for sampling.
        valid_js_values: Raw JS values over the complete valid set.
        valid_probabilities: Normalized sampling probabilities over the valid set.
        valid_token_count: Size of the valid set.
        sampled_token_count: Number of selected positions.
        seed: Fully derived replay seed supplied to PyTorch.
        global_step: Optimizer step included in seed derivation.
        sample_id: Stable string form of the sample identifier.
        rank: Distributed rank included in seed derivation.
    """

    selected_positions: tuple[int, ...]
    selected_js_values: tuple[float, ...]
    valid_positions: tuple[int, ...]
    valid_js_values: tuple[float, ...]
    valid_probabilities: tuple[float, ...]
    valid_token_count: int
    sampled_token_count: int
    seed: int
    global_step: int
    sample_id: str
    rank: int


def sample_token_positions(
    token_js: Any,
    valid_mask: Any,
    config: JSTokenSamplingConfig,
    *,
    global_step: int,
    sample_id: str | int,
    rank: int = 0,
) -> TokenSampleSelection:
    """Sample positions from a uniform/JS mixture without retaining gradients.

    Args:
        token_js: One-dimensional per-position JS tensor or tensor-like value.
        valid_mask: Boolean mask aligned with ``token_js``.
        config: Validated sampling policy.
        global_step: Non-negative optimizer step used for replay seeding.
        sample_id: Stable sample identifier used for replay seeding.
        rank: Non-negative distributed rank or replica identifier.

    Returns:
        A CPU/primitive ``TokenSampleSelection`` suitable for metrics and transport.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If inputs are misaligned, no valid token exists, metadata is
            negative, or valid JS/sampling weights are non-finite or negative.

    Example:
        ``sample = sample_token_positions(js, mask, config, global_step=2, sample_id="a")``
    """
    torch = _torch()
    js = torch.as_tensor(token_js).detach().float().cpu()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool).detach().cpu()
    if js.ndim != 1 or mask.ndim != 1 or js.shape != mask.shape:
        raise ValueError("token_js and valid_mask must be aligned one-dimensional tensors")
    if global_step < 0 or rank < 0:
        raise ValueError("global_step and rank must be non-negative")
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("Gram sampling requires at least one valid token")
    valid_js = js[positions]
    if not torch.isfinite(valid_js).all() or (valid_js < 0).any():
        raise ValueError("valid JS scores must be finite and non-negative")
    powered = (valid_js + config.minimum_weight).pow(config.divergence_power)
    if not torch.isfinite(powered).all() or powered.sum().item() <= 0.0:
        raise ValueError("JS sampling weights must have finite positive mass")
    uniform = torch.full_like(powered, 1.0 / powered.numel())
    prioritized = powered / powered.sum()
    probabilities = (1.0 - config.js_sampling_mix) * uniform + config.js_sampling_mix * prioritized
    probabilities = probabilities / probabilities.sum()

    derived_seed = derive_sampling_seed(
        config.seed,
        global_step=global_step,
        sample_id=sample_id,
        rank=rank,
    )
    sample_count = min(config.sample_size, positions.numel())
    if sample_count == positions.numel() and config.sample_without_replacement:
        selected = positions
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derived_seed)
        selected_indices = torch.multinomial(
            probabilities,
            sample_count,
            replacement=not config.sample_without_replacement,
            generator=generator,
        )
        selected = positions[selected_indices]
    selected_js = js[selected]
    return TokenSampleSelection(
        selected_positions=tuple(int(value) for value in selected.tolist()),
        selected_js_values=tuple(float(value) for value in selected_js.tolist()),
        valid_positions=tuple(int(value) for value in positions.tolist()),
        valid_js_values=tuple(float(value) for value in valid_js.tolist()),
        valid_probabilities=tuple(float(value) for value in probabilities.tolist()),
        valid_token_count=int(positions.numel()),
        sampled_token_count=int(selected.numel()),
        seed=derived_seed,
        global_step=global_step,
        sample_id=str(sample_id),
        rank=rank,
    )


def derive_sampling_seed(
    configured_seed: int, *, global_step: int, sample_id: str | int, rank: int
) -> int:
    """Derive a stable PyTorch seed independent of Python hash randomization.

    Args:
        configured_seed: User-provided base seed.
        global_step: Optimizer step.
        sample_id: Stable sample identifier.
        rank: Distributed rank or replica identifier.

    Returns:
        A deterministic non-negative integer below ``2**63 - 1``.
    """
    payload = f"{configured_seed}:{global_step}:{sample_id}:{rank}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**63 - 1)


def validate_sampling_probabilities(probabilities: tuple[float, ...]) -> None:
    """Validate serialized selection probabilities for transports and caches.

    Args:
        probabilities: Non-empty categorical probabilities expected to sum to one.

    Raises:
        ValueError: If the tuple is empty, contains invalid values, or lacks unit mass.
    """
    if not probabilities:
        raise ValueError("sampling probabilities must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("sampling probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("sampling probabilities must sum to one")


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Gram regularization tensors") from exc


__all__ = [
    "TokenSampleSelection",
    "derive_sampling_seed",
    "sample_token_positions",
    "validate_sampling_probabilities",
]
