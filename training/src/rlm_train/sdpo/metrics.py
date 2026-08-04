"""Accumulate SDPO component losses and subcall information-significance metrics.

Purpose:
    Keep dense distillation diagnostics separate from the existing final-answer reward.
Implementation:
    Small mutable dataclasses aggregate token-normalized loss numerators and signed
    information significance without depending on a metrics backend.
Inputs:
    Per-component loss totals, active-token counts, and signed subcall scores.
Outputs:
    Optional arithmetic means suitable for logging to Prime or another tracker.
Example:
    ``metric = SubcallInformationMetric(); metric.add(0.8)``
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComponentMetric:
    """Accumulate a component loss numerator and active-token denominator.

    Attributes:
        loss_sum: Sum of loss values over active tokens.
        active_tokens: Number of active tokens represented by ``loss_sum``.
    """

    loss_sum: float = 0.0
    active_tokens: int = 0

    @property
    def mean_loss(self) -> float | None:
        """Return mean loss per active token, or ``None`` when no token is active."""
        if self.active_tokens == 0:
            return None
        return self.loss_sum / self.active_tokens


@dataclass
class SubcallInformationMetric:
    """Accumulate signed information significance independently of answer quality.

    Attributes:
        significance_sum: Sum of accepted values in ``[-1, 1]``.
        subcall_count: Number of assessed subcalls.
    """

    significance_sum: float = 0.0
    subcall_count: int = 0

    def add(self, significance: float) -> None:
        """Add one signed information-value score.

        Args:
            significance: Reward/penalty placeholder in the inclusive range ``[-1, 1]``.

        Raises:
            ValueError: If ``significance`` lies outside the schema range.
        """
        if significance < -1.0 or significance > 1.0:
            raise ValueError("subcall information significance must be between -1 and 1")
        self.significance_sum += significance
        self.subcall_count += 1

    @property
    def mean_significance(self) -> float | None:
        """Return mean significance, or ``None`` before any subcall is assessed."""
        if self.subcall_count == 0:
            return None
        return self.significance_sum / self.subcall_count
