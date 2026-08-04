"""Build structured diagnostics for Gram loss, JS drift, sampling, and anchor age.

Purpose:
    Make prioritized-sampling behavior and representation drift observable independently
    of the policy and SDPO metrics backends.
Implementation:
    Immutable dataclasses store summaries; helper functions compute deterministic JS
    quantiles, scalarize detached tensor losses, and flatten records into stable keys.
Inputs:
    Gram loss results, token selections, resolved layers, anchor identity, sampling
    mixture values, optimizer step, and optional decision-component masks.
Outputs:
    ``GramAnchorMetrics`` records and tracker-ready dictionaries.
Example:
    ``metrics = build_gram_anchor_metrics(loss, sample, layers, anchor_id, global_step=10, global_loss_weight=0.1, js_sampling_mix=0.8)``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlm_train.regularization.anchor import AnchorIdentity
from rlm_train.regularization.gram import GramAnchorLossResult
from rlm_train.regularization.sampling import TokenSampleSelection
from rlm_train.regularization.selectors import ResolvedLayerSelection


@dataclass(frozen=True)
class JSSummary:
    """Summarize a finite, non-empty set of token JS scores.

    Attributes:
        mean: Arithmetic mean across positions.
        maximum: Largest observed JS value.
        q50: Median from deterministic linear interpolation.
        q90: Ninetieth percentile.
        q99: Ninety-ninth percentile.
    """

    mean: float
    maximum: float
    q50: float
    q90: float
    q99: float


@dataclass(frozen=True)
class GramAnchorMetrics:
    """Carry framework-neutral values suitable for any metrics backend.

    Attributes:
        weighted_gram_loss: Global auxiliary loss applied to the objective.
        unweighted_gram_loss: Layer-aggregated ordinary Gram diagnostic.
        per_layer_gram_losses: Optimized losses keyed by resolved block index.
        per_layer_unweighted_gram_losses: Ordinary losses keyed by block index.
        global_loss_weight: Configured global objective coefficient.
        per_layer_effective_weights: Actual global coefficient per block.
        valid_js: JS summary across all eligible token positions.
        sampled_js: JS summary across selected positions.
        valid_token_count: Number of positions eligible for sampling.
        sampled_token_count: Number of positions used in Gram matrices.
        uniform_sampling_mix: Uniform probability-mixture coefficient.
        js_sampling_mix: JS-prioritized probability-mixture coefficient.
        requested_relative_depths: Original relative layer policy, when used.
        resolved_layer_indices: Frozen zero-based transformer block indices.
        anchor_identifier: Stable anchor checkpoint or logical name.
        anchor_version: Monotonic anchor version.
        anchor_age: Optimizer steps since the anchor version was created.
        sampled_decision_fractions: Optional sampled fraction by decision component.
    """

    weighted_gram_loss: float
    unweighted_gram_loss: float
    per_layer_gram_losses: dict[int, float]
    per_layer_unweighted_gram_losses: dict[int, float]
    global_loss_weight: float
    per_layer_effective_weights: dict[int, float]
    valid_js: JSSummary
    sampled_js: JSSummary
    valid_token_count: int
    sampled_token_count: int
    uniform_sampling_mix: float
    js_sampling_mix: float
    requested_relative_depths: tuple[float, ...] | None
    resolved_layer_indices: tuple[int, ...]
    anchor_identifier: str
    anchor_version: int
    anchor_age: int
    sampled_decision_fractions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Flatten nested summaries into stable tracker keys.

        Returns:
            A new dictionary with ``gram/``-prefixed scalar, list, and metadata values.
        """
        result: dict[str, Any] = {
            "gram/loss": self.weighted_gram_loss,
            "gram/unweighted_loss": self.unweighted_gram_loss,
            "gram/loss_weight": self.global_loss_weight,
            "gram/valid_token_count": self.valid_token_count,
            "gram/sampled_token_count": self.sampled_token_count,
            "gram/uniform_sampling_mix": self.uniform_sampling_mix,
            "gram/js_sampling_mix": self.js_sampling_mix,
            "gram/resolved_layer_indices": list(self.resolved_layer_indices),
            "gram/requested_relative_depths": (
                list(self.requested_relative_depths)
                if self.requested_relative_depths is not None
                else None
            ),
            "gram/anchor_identifier": self.anchor_identifier,
            "gram/anchor_version": self.anchor_version,
            "gram/anchor_age": self.anchor_age,
        }
        for prefix, summary in (("valid_js", self.valid_js), ("sampled_js", self.sampled_js)):
            for key, value in summary.__dict__.items():
                result[f"gram/{prefix}/{key}"] = value
        for layer, value in self.per_layer_gram_losses.items():
            result[f"gram/layer/{layer}/loss"] = value
            result[f"gram/layer/{layer}/unweighted_loss"] = self.per_layer_unweighted_gram_losses[
                layer
            ]
            result[f"gram/layer/{layer}/effective_weight"] = self.per_layer_effective_weights[layer]
        for component, fraction in self.sampled_decision_fractions.items():
            result[f"gram/sampled_component/{component}"] = fraction
        return result


def summarize_js(values: Sequence[float]) -> JSSummary:
    """Compute deterministic JS summary statistics.

    Args:
        values: Non-empty finite, non-negative JS values.

    Returns:
        Mean, maximum, median, ninetieth, and ninety-ninth percentiles.

    Raises:
        ValueError: If values are empty, negative, or non-finite.
    """
    numeric = np.asarray(values, dtype=np.float64)
    if (
        numeric.ndim != 1
        or numeric.size == 0
        or not np.isfinite(numeric).all()
        or (numeric < 0.0).any()
    ):
        raise ValueError("JS summaries require finite, non-negative values")
    q50, q90, q99 = np.quantile(numeric, (0.50, 0.90, 0.99), method="linear")
    return JSSummary(
        mean=float(numeric.mean()),
        maximum=float(numeric.max()),
        q50=float(q50),
        q90=float(q90),
        q99=float(q99),
    )


def build_gram_anchor_metrics(
    loss: GramAnchorLossResult,
    sample: TokenSampleSelection,
    selection: ResolvedLayerSelection,
    anchor: AnchorIdentity,
    *,
    global_step: int,
    global_loss_weight: float,
    js_sampling_mix: float,
    decision_masks: Mapping[str, Sequence[bool]] | None = None,
) -> GramAnchorMetrics:
    """Combine objective and selection diagnostics into one metrics record.

    Args:
        loss: Multi-layer Gram loss result.
        sample: Reproducible token-selection record.
        selection: Frozen resolved layer metadata.
        anchor: Anchor identity used for logits and hidden states.
        global_step: Current optimizer step used to calculate anchor age.
        global_loss_weight: Configured global Gram coefficient.
        js_sampling_mix: JS-prioritized sampling mixture in ``[0, 1]``.
        decision_masks: Optional full-sequence component masks.

    Returns:
        A detached, framework-neutral ``GramAnchorMetrics`` record.

    Raises:
        ValueError: If the sampling mixture is invalid, component masks do not cover
            sampled positions, or JS summaries contain invalid values.
    """
    if js_sampling_mix < 0.0 or js_sampling_mix > 1.0:
        raise ValueError("js_sampling_mix must be in [0, 1]")
    fractions: dict[str, float] = {}
    for component, mask in (decision_masks or {}).items():
        if len(mask) <= max(sample.selected_positions):
            raise ValueError("decision masks must cover every sampled position")
        fractions[component] = (
            sum(bool(mask[position]) for position in sample.selected_positions)
            / sample.sampled_token_count
        )
    return GramAnchorMetrics(
        weighted_gram_loss=_scalar(loss.total_loss),
        unweighted_gram_loss=_scalar(loss.unweighted_diagnostic_loss),
        per_layer_gram_losses={index: _scalar(value) for index, value in loss.layer_losses.items()},
        per_layer_unweighted_gram_losses={
            index: _scalar(value) for index, value in loss.unweighted_layer_losses.items()
        },
        global_loss_weight=global_loss_weight,
        per_layer_effective_weights=dict(loss.effective_layer_weights),
        valid_js=summarize_js(sample.valid_js_values),
        sampled_js=summarize_js(sample.selected_js_values),
        valid_token_count=sample.valid_token_count,
        sampled_token_count=sample.sampled_token_count,
        uniform_sampling_mix=1.0 - js_sampling_mix,
        js_sampling_mix=js_sampling_mix,
        requested_relative_depths=selection.requested_relative_depths,
        resolved_layer_indices=selection.indices,
        anchor_identifier=anchor.identifier,
        anchor_version=anchor.version,
        anchor_age=anchor.age(global_step),
        sampled_decision_fractions=fractions,
    )


def _scalar(value: Any) -> float:
    """Convert tensors or numeric values without preserving autograd references."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


__all__ = [
    "GramAnchorMetrics",
    "JSSummary",
    "build_gram_anchor_metrics",
    "summarize_js",
]
