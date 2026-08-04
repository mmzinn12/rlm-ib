"""Validate framework-neutral Gram policy and deterministic layer resolution.

Purpose:
    Protect configuration defaults, activation rules, ranges, and layer-resolution
    invariants without requiring a model or tensor runtime.
Implementation:
    Pydantic construction tests cover valid defaults and expected validation failures;
    pure selector tests resolve explicit and relative layer policies.
Inputs:
    Synthetic configuration values and transformer block counts.
Outputs:
    Pytest assertions and expected ``ValidationError`` instances.
Example:
    Run ``pytest training/tests/test_regularization_config.py`` from the repository root.
"""

import pytest
from pydantic import ValidationError

from rlm_train.regularization.config import (
    GramAnchorConfig,
    GramAnchorSourceConfig,
    GramLayerSelectionConfig,
    JSTokenSamplingConfig,
)
from rlm_train.regularization.selectors import resolve_layer_selection


def test_default_relative_layers_resolve_once_at_locked_depths():
    resolved = resolve_layer_selection(GramLayerSelectionConfig(), block_count=32)

    assert resolved.indices == (15, 23, 31)
    assert resolved.weights == (1.0, 1.0, 1.0)
    assert resolve_layer_selection(GramLayerSelectionConfig(), block_count=32) == resolved


def test_explicit_indices_override_default_relative_depths():
    config = GramLayerSelectionConfig(indices=(2, 5), layer_weights=(1.0, 3.0))

    assert config.relative_depths is None
    assert resolve_layer_selection(config, block_count=8).indices == (2, 5)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: JSTokenSamplingConfig(sample_size=0),
        lambda: JSTokenSamplingConfig(top_k=0),
        lambda: JSTokenSamplingConfig(js_sampling_mix=1.1),
        lambda: GramLayerSelectionConfig(indices=(1, 1)),
        lambda: GramLayerSelectionConfig(relative_depths=(0.0,)),
        lambda: GramLayerSelectionConfig(relative_depths=(0.5, 1.0), layer_weights=(1.0,)),
        lambda: GramAnchorSourceConfig(strategy="periodic_ema_snapshot", update_interval=0),
    ],
)
def test_invalid_gram_configuration_fails_loudly(factory):
    with pytest.raises(ValidationError):
        factory()


def test_active_fixed_anchor_requires_a_checkpoint_and_positive_layer_weight():
    with pytest.raises(ValidationError, match="checkpoint_path"):
        GramAnchorConfig(enabled=True, loss_weight=1.0)
    with pytest.raises(ValidationError, match="positive layer weight"):
        GramAnchorConfig(
            enabled=True,
            loss_weight=1.0,
            anchor=GramAnchorSourceConfig(checkpoint_path="base"),
            layers=GramLayerSelectionConfig(layer_weights=(0.0, 0.0, 0.0)),
        )


def test_zero_loss_weight_disables_objective_without_loading_a_checkpoint():
    assert GramAnchorConfig(enabled=True, loss_weight=0.0).is_active is False
