"""Expose the public, SDPO-independent Gram-anchor regularization API.

Purpose:
    Provide one stable import surface for configuration, anchor lifecycle, JS drift,
    token selection, Gram losses, diagnostics, and trainer integration types.
Implementation:
    This facade re-exports framework-neutral symbols from the regularization package and
    performs no model loading, tensor allocation, sampling, or optimization itself.
Inputs:
    Python imports from training configuration, model adapters, or tests.
Outputs:
    Public Gram-anchor classes, protocols, functions, and transport dataclasses.
Example:
    ``from rlm_train.regularization import GramAnchorConfig, gram_matrix_loss``
"""

from rlm_train.regularization.anchor import (
    AlignedAnchorSource,
    AlignedSequenceInputs,
    AnchorForwardOutput,
    AnchorIdentity,
    FixedCheckpointAnchorController,
    PeriodicEMASnapshotAnchorController,
)
from rlm_train.regularization.config import (
    GramAnchorConfig,
    GramAnchorSourceConfig,
    GramLayerSelectionConfig,
    JSTokenSamplingConfig,
)
from rlm_train.regularization.divergence import (
    CoarsenedDistributions,
    ReferenceLogitSource,
    coarsen_logits_to_reference_topk,
    per_token_js_divergence,
)
from rlm_train.regularization.gram import (
    GramAnchorLossResult,
    GramLayerLoss,
    gram_matrix_loss,
    multi_layer_gram_loss,
)
from rlm_train.regularization.metrics import GramAnchorMetrics, JSSummary
from rlm_train.regularization.prime_adapter import (
    GramAnchorPrimeAdapter,
    PrimeGramAnchorFields,
    PrimeGramRegularizationStep,
    SelectedLayerHookCapture,
    compose_training_objective,
)
from rlm_train.regularization.sampling import TokenSampleSelection, sample_token_positions
from rlm_train.regularization.selectors import (
    ResolvedLayerSelection,
    build_completion_mask,
    build_valid_token_mask,
    resolve_layer_selection,
)

__all__ = [
    "AlignedAnchorSource",
    "AlignedSequenceInputs",
    "AnchorForwardOutput",
    "AnchorIdentity",
    "CoarsenedDistributions",
    "FixedCheckpointAnchorController",
    "GramAnchorConfig",
    "GramAnchorLossResult",
    "GramAnchorMetrics",
    "GramAnchorPrimeAdapter",
    "GramAnchorSourceConfig",
    "GramLayerLoss",
    "GramLayerSelectionConfig",
    "JSSummary",
    "JSTokenSamplingConfig",
    "PeriodicEMASnapshotAnchorController",
    "PrimeGramAnchorFields",
    "PrimeGramRegularizationStep",
    "ReferenceLogitSource",
    "ResolvedLayerSelection",
    "SelectedLayerHookCapture",
    "TokenSampleSelection",
    "build_completion_mask",
    "build_valid_token_mask",
    "coarsen_logits_to_reference_topk",
    "compose_training_objective",
    "gram_matrix_loss",
    "multi_layer_gram_loss",
    "per_token_js_divergence",
    "resolve_layer_selection",
    "sample_token_positions",
]
