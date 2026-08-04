"""Compute observer-only reasoning diagnostics without affecting model inputs or losses."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rlm.core.trajectory import TrajectoryTree

EPISTEMIC_MARKERS = (
    "wait",
    "hmm",
    "perhaps",
    "maybe",
    "actually",
    "alternatively",
    "seems",
    "might",
    "likely",
    "check",
)


class ImmutableDiagnostic(BaseModel):
    """Reject unknown values and freeze diagnostic records after observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EpistemicMarkerSummary(ImmutableDiagnostic):
    """Store marker counts and rates normalized by observed response tokens."""

    counts: dict[str, int]
    rates: dict[str, float]
    total_count: int = Field(ge=0)


class ReasoningDynamics(ImmutableDiagnostic):
    """Count textual reconsideration, verification, and correction signals."""

    abandoned_approach_count: int = Field(ge=0)
    derived_result_check_count: int = Field(ge=0)
    correction_count: int = Field(ge=0)
    revision_count: int = Field(ge=0)
    revision_rate: float = Field(ge=0.0)


class TrajectoryDynamics(ImmutableDiagnostic):
    """Summarize recursive call topology and runtime retries."""

    question_count: int = Field(ge=0)
    subcall_count: int = Field(ge=0)
    maximum_depth: int = Field(ge=0)
    maximum_breadth: int = Field(ge=0)
    retry_count: int = Field(ge=0)


class DivergenceSummary(ImmutableDiagnostic):
    """Summarize teacher-student divergence globally and at epistemic positions."""

    mean: float = Field(ge=0.0)
    maximum: float = Field(ge=0.0)
    epistemic_mean: float | None = Field(default=None, ge=0.0)
    epistemic_position_count: int = Field(ge=0)


class ObserverDiagnostics(ImmutableDiagnostic):
    """Carry all response observations with no reward or objective fields."""

    response_token_count: int = Field(ge=0)
    truncated: bool
    epistemic: EpistemicMarkerSummary
    reasoning: ReasoningDynamics
    trajectory: TrajectoryDynamics | None = None
    divergence: DivergenceSummary | None = None
    gram: dict[str, Any] | None = None


def collect_observer_diagnostics(
    response: str,
    *,
    response_tokens: Sequence[str] | None = None,
    token_count: int | None = None,
    truncated: bool = False,
    trajectory: TrajectoryTree | None = None,
    per_token_divergence: Sequence[float] | None = None,
    gram_metrics: Mapping[str, Any] | None = None,
) -> ObserverDiagnostics:
    """Observe a completed response without mutating any supplied object."""
    tokens = tuple(response_tokens) if response_tokens is not None else tuple(response.split())
    resolved_token_count = len(tokens) if token_count is None else token_count
    if resolved_token_count < 0:
        raise ValueError("response token count must be non-negative")
    if response_tokens is not None and token_count is not None and len(tokens) != token_count:
        raise ValueError("response_tokens and token_count must agree")
    marker_counts = {
        marker: len(re.findall(rf"\b{re.escape(marker)}\b", response, flags=re.IGNORECASE))
        for marker in EPISTEMIC_MARKERS
    }
    denominator = max(resolved_token_count, 1)
    epistemic = EpistemicMarkerSummary(
        counts=marker_counts,
        rates={marker: count / denominator for marker, count in marker_counts.items()},
        total_count=sum(marker_counts.values()),
    )
    abandoned = _count_patterns(
        response,
        (r"\babandon(?:ing|ed)?\b", r"\banother approach\b", r"\binstead[, :]"),
    )
    checked = _count_patterns(
        response,
        (r"\bcheck(?:ing|ed)?\b", r"\bverif(?:y|ying|ied)\b", r"\brecomput(?:e|ing|ed)\b"),
    )
    corrected = _count_patterns(
        response,
        (r"\bactually\b", r"\bcorrection\b", r"\bi was wrong\b", r"\bthat was wrong\b"),
    )
    revisions = abandoned + checked + corrected
    reasoning = ReasoningDynamics(
        abandoned_approach_count=abandoned,
        derived_result_check_count=checked,
        correction_count=corrected,
        revision_count=revisions,
        revision_rate=revisions / denominator,
    )
    divergence = _summarize_divergence(per_token_divergence, tokens)
    return ObserverDiagnostics(
        response_token_count=resolved_token_count,
        truncated=truncated,
        epistemic=epistemic,
        reasoning=reasoning,
        trajectory=_summarize_trajectory(trajectory) if trajectory is not None else None,
        divergence=divergence,
        gram=dict(gram_metrics) if gram_metrics is not None else None,
    )


def effective_rank_from_singular_values(values: Sequence[float]) -> float:
    """Compute entropy effective rank from detached non-negative singular values."""
    singular_values = tuple(float(value) for value in values)
    if not singular_values:
        raise ValueError("effective rank requires at least one singular value")
    if any(value < 0.0 or not math.isfinite(value) for value in singular_values):
        raise ValueError("singular values must be finite and non-negative")
    total = sum(singular_values)
    if total == 0.0:
        return 0.0
    probabilities = (value / total for value in singular_values if value > 0.0)
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return math.exp(entropy)


def build_gram_observer_metrics(
    *,
    gram_drift: float,
    per_layer_gram_losses: Mapping[int, float],
    per_layer_singular_values: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    """Build detached Gram drift, per-layer loss, and effective-rank observations."""
    drift = float(gram_drift)
    losses = {int(layer): float(value) for layer, value in per_layer_gram_losses.items()}
    if drift < 0.0 or not math.isfinite(drift):
        raise ValueError("Gram drift must be finite and non-negative")
    if any(value < 0.0 or not math.isfinite(value) for value in losses.values()):
        raise ValueError("per-layer Gram losses must be finite and non-negative")
    if set(losses) != set(per_layer_singular_values):
        raise ValueError("Gram losses and singular values must cover identical layers")
    return {
        "gram_drift": drift,
        "per_layer_gram_loss": losses,
        "per_layer_effective_rank": {
            int(layer): effective_rank_from_singular_values(values)
            for layer, values in per_layer_singular_values.items()
        },
    }


def _count_patterns(text: str, patterns: Sequence[str]) -> int:
    """Count non-overlapping matches from a small explicit behavioral lexicon."""
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def _summarize_trajectory(trajectory: TrajectoryTree) -> TrajectoryDynamics:
    """Read topology and retry metadata from a validated trajectory."""
    trajectory.validate()
    breadth: Counter[str] = Counter()
    for node in trajectory.nodes:
        if node.parent_id is not None:
            breadth[node.parent_id] += 1
    question_count = sum(len(node.call_item_spans) for node in trajectory.nodes)
    retry_count = 0
    for node in trajectory.nodes:
        for key in ("retry_count", "retries"):
            value = node.metadata.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"trajectory metadata {key!r} must be a non-negative integer")
            retry_count += value
    return TrajectoryDynamics(
        question_count=question_count,
        subcall_count=sum(node.parent_id is not None for node in trajectory.nodes),
        maximum_depth=max((node.depth for node in trajectory.nodes), default=0),
        maximum_breadth=max(breadth.values(), default=0),
        retry_count=retry_count,
    )


def _summarize_divergence(
    values: Sequence[float] | None,
    tokens: Sequence[str],
) -> DivergenceSummary | None:
    """Validate aligned detached values and isolate epistemic-token positions."""
    if values is None:
        return None
    numeric = tuple(float(value) for value in values)
    if len(numeric) != len(tokens):
        raise ValueError("per-token divergence must align with response_tokens")
    if not numeric:
        raise ValueError("per-token divergence must not be empty")
    if any(value < 0.0 or not math.isfinite(value) for value in numeric):
        raise ValueError("per-token divergence must contain finite non-negative values")
    epistemic_values = [
        value
        for token, value in zip(tokens, numeric, strict=True)
        if token.casefold().strip(".,:;!?()[]{}\"'") in EPISTEMIC_MARKERS
    ]
    return DivergenceSummary(
        mean=sum(numeric) / len(numeric),
        maximum=max(numeric),
        epistemic_mean=(
            sum(epistemic_values) / len(epistemic_values) if epistemic_values else None
        ),
        epistemic_position_count=len(epistemic_values),
    )


__all__ = [
    "DivergenceSummary",
    "EPISTEMIC_MARKERS",
    "EpistemicMarkerSummary",
    "ObserverDiagnostics",
    "ReasoningDynamics",
    "TrajectoryDynamics",
    "build_gram_observer_metrics",
    "collect_observer_diagnostics",
    "effective_rank_from_singular_values",
]
