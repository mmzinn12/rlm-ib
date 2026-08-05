"""Define immutable configuration for modular Gram-anchor regularization.

Purpose:
    Encode stable anchor, JS sampling, layer selection, and loss policies separately
    from model-specific loading and trainer behavior.
Implementation:
    Frozen Pydantic models reject unknown fields and validate strategy combinations,
    ranges, selection modes, checkpoint requirements, and objective activation.
Inputs:
    User configuration values parsed from Python, TOML, JSON, or another Pydantic source.
Outputs:
    Validated ``GramAnchorConfig`` and nested immutable policy models.
Example:
    ``config = GramAnchorConfig(enabled=True, loss_weight=0.1, anchor={"checkpoint_path": "base"})``
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rlm_train.spec.objectives import GramSpec


class ImmutableConfig(BaseModel):
    """Base model that rejects unknown fields and freezes validated values.

    Subclasses inherit Pydantic's ``extra="forbid"`` and ``frozen=True`` behavior so
    experiment policy cannot drift through misspelled or post-construction mutations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class GramAnchorSourceConfig(ImmutableConfig):
    """Describe the independently versioned representation-anchor lifecycle.

    Attributes:
        strategy: Fixed pre-training checkpoint or periodic EMA snapshot policy.
        checkpoint_path: Optional stable source checkpoint; required for an active fixed
            anchor by ``GramAnchorConfig``.
        update_interval: Positive refresh interval required only for periodic snapshots.

    Example:
        ``GramAnchorSourceConfig(checkpoint_path="checkpoints/pre-sdpo")``
    """

    strategy: Literal["fixed_checkpoint", "periodic_ema_snapshot"] = "fixed_checkpoint"
    checkpoint_path: str | None = None
    update_interval: int | None = None

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> GramAnchorSourceConfig:
        """Validate lifecycle-specific fields.

        Returns:
            This validated immutable configuration.

        Raises:
            ValueError: If periodic snapshots lack a positive interval, a fixed anchor
                specifies an interval, or the checkpoint path is blank.
        """
        if self.strategy == "periodic_ema_snapshot":
            if self.update_interval is None or self.update_interval <= 0:
                raise ValueError("periodic EMA snapshots require a positive update_interval")
        elif self.update_interval is not None:
            raise ValueError("fixed checkpoints do not accept update_interval")
        if self.checkpoint_path is not None and not self.checkpoint_path.strip():
            raise ValueError("checkpoint_path must not be blank")
        return self


class JSTokenSamplingConfig(ImmutableConfig):
    """Configure detached JS scoring, token scope, and bounded sampling.

    Attributes:
        reference_source: Aligned source used only for detached JS logits.
        vocabulary_support: Full vocabulary or reference top-k plus tail.
        top_k: Positive explicit support width for top-k coarsening.
        token_scope: Completion, all attended tokens, or a supplied decision mask.
        sample_size: Maximum positions selected per sample.
        js_sampling_mix: Mixture weight assigned to JS-prioritized probabilities.
        divergence_power: Exponent applied to non-negative JS scores.
        minimum_weight: Positive epsilon preventing zero sampling/pair weight.
        sample_without_replacement: Whether selected positions must be unique.
        seed: Base seed combined with sample and distributed metadata.

    Example:
        ``sampling = JSTokenSamplingConfig(sample_size=256, js_sampling_mix=0.7)``
    """

    reference_source: Literal["gram_anchor", "ema_teacher", "custom"] = "gram_anchor"
    vocabulary_support: Literal["reference_topk_tail", "full"] = "reference_topk_tail"
    top_k: int = Field(default=100, gt=0)
    token_scope: Literal["completion", "all_valid", "decision"] = "completion"
    sample_size: int = Field(default=512, gt=0)
    js_sampling_mix: float = Field(default=0.8, ge=0.0, le=1.0)
    divergence_power: float = Field(default=1.0, ge=0.0)
    minimum_weight: float = Field(default=1e-8, gt=0.0)
    sample_without_replacement: bool = True
    seed: int = 0


class GramLayerSelectionConfig(ImmutableConfig):
    """Select transformer blocks by absolute index or relative depth.

    Attributes:
        indices: Optional unique zero-based block indices; overrides default depths.
        relative_depths: Optional depths in ``(0, 1]`` resolved at model initialization.
        layer_weights: Optional non-negative weights aligned with the requested layers.

    Raises:
        ValueError: If both/neither selection modes are supplied, selections are empty
            or invalid, or layer weights do not align.

    Example:
        ``layers = GramLayerSelectionConfig(indices=(15, 23, 31))``
    """

    indices: tuple[int, ...] | None = None
    relative_depths: tuple[float, ...] | None = (0.50, 0.75, 1.00)
    layer_weights: tuple[float, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def explicit_indices_override_defaults(cls, data: object) -> object:
        """Disable default relative depths when explicit indices are supplied.

        Args:
            data: Raw Pydantic input before field parsing.

        Returns:
            A copied mapping with ``relative_depths=None`` when appropriate, otherwise
            the original input.
        """
        if isinstance(data, dict) and data.get("indices") is not None:
            values = dict(data)
            values.setdefault("relative_depths", None)
            return values
        return data

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        """Validate explicit block indices.

        Args:
            value: Optional requested zero-based indices.

        Returns:
            The unchanged validated tuple or ``None``.

        Raises:
            ValueError: If the tuple is empty, negative, or contains duplicates.
        """
        if value is None:
            return value
        if not value:
            raise ValueError("indices must contain at least one layer")
        if any(index < 0 for index in value):
            raise ValueError("layer indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("layer indices must be unique")
        return value

    @field_validator("relative_depths")
    @classmethod
    def validate_relative_depths(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        """Validate relative-depth requests.

        Args:
            value: Optional requested depths.

        Returns:
            The unchanged validated tuple or ``None``.

        Raises:
            ValueError: If the tuple is empty or any depth lies outside ``(0, 1]``.
        """
        if value is None:
            return value
        if not value:
            raise ValueError("relative_depths must contain at least one layer")
        if any(depth <= 0.0 or depth > 1.0 for depth in value):
            raise ValueError("relative layer depths must be in (0, 1]")
        return value

    @model_validator(mode="after")
    def validate_selection_and_weights(self) -> GramLayerSelectionConfig:
        """Validate selection exclusivity and requested layer weights.

        Returns:
            This validated immutable configuration.

        Raises:
            ValueError: If selection modes are ambiguous, weights have the wrong count,
                or a weight is negative.
        """
        if (self.indices is None) == (self.relative_depths is None):
            raise ValueError("exactly one of indices or relative_depths must be supplied")
        requested_count = len(self.indices or self.relative_depths or ())
        if self.layer_weights is not None:
            if len(self.layer_weights) != requested_count:
                raise ValueError("layer_weights must align with the selected layers")
            if any(weight < 0.0 for weight in self.layer_weights):
                raise ValueError("layer weights must be non-negative")
        return self


class GramAnchorConfig(ImmutableConfig):
    """Configure the auxiliary objective independently of token-level SDPO.

    Attributes:
        enabled: Explicit feature gate for anchor forward/loss work.
        loss_weight: Non-negative global coefficient; zero disables computation.
        pair_weighting: JS-product weighting or ordinary unweighted Gram MSE.
        normalize_hidden_states: Whether to L2-normalize selected token features.
        anchor: Anchor source and lifecycle policy.
        sampling: JS reference, token scope, and sampling policy.
        layers: Requested layers and aggregation weights.

    Example:
        ``config = GramAnchorConfig(enabled=True, loss_weight=0.1, anchor={"checkpoint_path": "base"})``
    """

    enabled: bool = False
    loss_weight: float = Field(default=0.0, ge=0.0)
    pair_weighting: Literal["js_product", "none"] = "js_product"
    normalize_hidden_states: bool = True
    anchor: GramAnchorSourceConfig = Field(default_factory=GramAnchorSourceConfig)
    sampling: JSTokenSamplingConfig = Field(default_factory=JSTokenSamplingConfig)
    layers: GramLayerSelectionConfig = Field(default_factory=GramLayerSelectionConfig)

    @property
    def is_active(self) -> bool:
        """Return whether the objective should perform forward and loss work."""
        return self.enabled and self.loss_weight > 0.0

    @model_validator(mode="after")
    def validate_active_run(self) -> GramAnchorConfig:
        """Validate requirements that apply only to an active objective.

        Returns:
            This validated immutable configuration.

        Raises:
            ValueError: If an active fixed anchor lacks a checkpoint or all explicit
                layer weights are zero.
        """
        if not self.is_active:
            return self
        if self.anchor.strategy == "fixed_checkpoint" and self.anchor.checkpoint_path is None:
            raise ValueError("an active fixed-checkpoint anchor requires checkpoint_path")
        if self.layers.layer_weights is not None and not any(
            weight > 0.0 for weight in self.layers.layer_weights
        ):
            raise ValueError("an active Gram objective requires a positive layer weight")
        return self


__all__ = [
    "GramAnchorConfig",
    "GramAnchorSourceConfig",
    "GramLayerSelectionConfig",
    "JSTokenSamplingConfig",
    "GramSpec",
]
