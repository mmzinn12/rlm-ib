"""Provide a thin selected-layer and objective seam for a pinned Prime trainer.

Purpose:
    Isolate stable Gram-anchor logic from version-specific Prime batch, model, hook, and
    objective APIs.
Implementation:
    Transport dataclasses carry aligned sample fields, selected-block hooks capture only
    requested student states, an adapter orchestrates anchor/JS/sampling/loss/metrics,
    and a pure helper composes independent training losses.
Inputs:
    Prime-aligned sample tensors, student logits and selected hidden states, a resolved
    configuration, an anchor source, optimizer step, and distributed rank.
Outputs:
    Optional ``PrimeGramRegularizationStep`` values and a combined scalar objective.
Example:
    ``step = adapter.compute(student_logits=logits, student_hidden_states=states, fields=fields, global_step=12)``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlm_train.regularization.anchor import AlignedAnchorSource, AlignedSequenceInputs
from rlm_train.regularization.config import GramAnchorConfig
from rlm_train.regularization.divergence import ReferenceLogitSource, per_token_js_divergence
from rlm_train.regularization.gram import GramAnchorLossResult, multi_layer_gram_loss
from rlm_train.regularization.metrics import GramAnchorMetrics, build_gram_anchor_metrics
from rlm_train.regularization.sampling import TokenSampleSelection, sample_token_positions
from rlm_train.regularization.selectors import (
    ResolvedLayerSelection,
    build_completion_mask,
    build_valid_token_mask,
)


@dataclass(frozen=True)
class PrimeGramAnchorFields:
    """Carry trainer-sample inputs without importing a Prime release.

    Attributes:
        sample_id: Stable identifier used in deterministic sampling.
        aligned_inputs: Feedback-free inputs shared by student and anchor.
        completion_mask: Token-aligned completion-position mask.
        special_position_mask: Optional true-for-special exclusion mask.
        decision_mask: Optional supplied mask for ``token_scope="decision"``.
        decision_component_masks: Optional masks used only for sampling diagnostics.

    Example:
        ``fields = PrimeGramAnchorFields("row-1", inputs, completion_mask)``
    """

    sample_id: str
    aligned_inputs: AlignedSequenceInputs
    completion_mask: Any
    special_position_mask: Any | None = None
    decision_mask: Any | None = None
    decision_component_masks: Mapping[str, Any] | None = None

    @classmethod
    def from_completion_starts(
        cls,
        *,
        sample_id: str,
        aligned_inputs: AlignedSequenceInputs,
        completion_start_positions: int | list[int] | Any,
        special_position_mask: Any | None = None,
        decision_mask: Any | None = None,
        decision_component_masks: Mapping[str, Any] | None = None,
    ) -> PrimeGramAnchorFields:
        """Construct fields and derive their completion mask.

        Args:
            sample_id: Stable identifier used for sampling replay.
            aligned_inputs: Inputs shared exactly by student and anchor.
            completion_start_positions: Scalar or per-row completion boundary.
            special_position_mask: Optional special-position exclusion mask.
            decision_mask: Optional decision-scope mask.
            decision_component_masks: Optional masks for component diagnostics.

        Returns:
            ``PrimeGramAnchorFields`` with a derived completion mask.

        Raises:
            RuntimeError: If PyTorch is unavailable.
            ValueError: If completion boundaries or masks are invalid.
        """
        completion = build_completion_mask(
            aligned_inputs.attention_mask,
            completion_start_positions,
            special_position_mask=special_position_mask,
        )
        return cls(
            sample_id=sample_id,
            aligned_inputs=aligned_inputs,
            completion_mask=completion,
            special_position_mask=special_position_mask,
            decision_mask=decision_mask,
            decision_component_masks=decision_component_masks,
        )


@dataclass(frozen=True)
class PrimeGramRegularizationStep:
    """Return one differentiable loss and its replay/logging records.

    Attributes:
        loss: Multi-layer Gram result whose ``total_loss`` retains student autograd.
        sample: Primitive replay metadata for selected token positions.
        metrics: Detached framework-neutral diagnostic values.
    """

    loss: GramAnchorLossResult
    sample: TokenSampleSelection
    metrics: GramAnchorMetrics


class SelectedLayerHookCapture:
    """Capture selected transformer-block outputs while preserving autograd.

    Args:
        blocks: Ordered transformer block modules supporting ``register_forward_hook``.
        layer_indices: Unique in-range block indices to capture.

    Attributes:
        hidden_states: Outputs keyed by selected block index after a model forward pass.

    Raises:
        ValueError: If an index is duplicated, negative, or outside ``blocks``.

    Example:
        ``with SelectedLayerHookCapture(blocks, (15, 23, 31)) as capture: model(**batch)``

    Hook handles are always removed on context exit. Captured student tensors are not
    detached, so gradients flow through the selected representation loss.
    """

    def __init__(self, blocks: Sequence[Any], layer_indices: tuple[int, ...]):
        if any(index < 0 or index >= len(blocks) for index in layer_indices):
            raise ValueError("selected hook layer exceeds available transformer blocks")
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError("selected hook layers must be unique")
        self._blocks = blocks
        self._layer_indices = layer_indices
        self._handles: list[Any] = []
        self.hidden_states: dict[int, Any] = {}

    def __enter__(self) -> SelectedLayerHookCapture:
        for index in self._layer_indices:
            self._handles.append(self._blocks[index].register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, index: int) -> Any:
        def capture(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            value = output[0] if isinstance(output, tuple) else output
            self.hidden_states[index] = value

        return capture


class GramAnchorPrimeAdapter:
    """Apply framework-neutral Gram pieces to one aligned Prime sample.

    Args:
        config: Validated Gram objective and sampling policy.
        selection: Layer indices resolved against the instantiated model.
        anchor: Feedback-free source of aligned logits and selected hidden states.
        reference_logit_source: Optional aligned non-anchor source used only for JS.

    Raises:
        ValueError: If an active selection has no positive weight or a configured
            non-anchor JS source is not supplied.

    Example:
        ``adapter = GramAnchorPrimeAdapter(config, layers, anchor)``
    """

    def __init__(
        self,
        config: GramAnchorConfig,
        selection: ResolvedLayerSelection,
        anchor: AlignedAnchorSource,
        *,
        reference_logit_source: ReferenceLogitSource | None = None,
    ):
        if config.is_active and not any(weight > 0.0 for weight in selection.weights):
            raise ValueError("an active Gram adapter requires a positive resolved layer weight")
        if config.sampling.reference_source != "gram_anchor" and reference_logit_source is None:
            raise ValueError("non-anchor JS reference_source requires an aligned source")
        self.config = config
        self.selection = selection
        self.anchor = anchor
        self.reference_logit_source = reference_logit_source

    def compute(
        self,
        *,
        student_logits: Any,
        student_hidden_states: Mapping[int, Any],
        fields: PrimeGramAnchorFields,
        global_step: int,
        rank: int = 0,
    ) -> PrimeGramRegularizationStep | None:
        """Compute one auxiliary loss and its diagnostics.

        Args:
            student_logits: Unbatched or singleton-batch aligned next-token logits.
            student_hidden_states: Selected student block outputs keyed by index.
            fields: Stable trainer-sample transport and token masks.
            global_step: Non-negative optimizer step used for sampling and anchor age.
            rank: Non-negative distributed rank used for deterministic sampling.

        Returns:
            A ``PrimeGramRegularizationStep`` or ``None`` when the objective is disabled.

        Raises:
            RuntimeError: If active tensor computation requires unavailable PyTorch.
            ValueError: If logits, masks, selected layers, token positions, reference
                source, or hidden states violate alignment and shape invariants.

        The anchor is teacher-forced on ``fields.aligned_inputs``. JS is detached before
        sampling, anchor states are detached, and student states retain autograd.
        """
        if not self.config.is_active:
            return None
        anchor_output = self.anchor.forward(fields.aligned_inputs, self.selection.indices)
        reference_logits = (
            anchor_output.logits
            if self.reference_logit_source is None
            else self.reference_logit_source.logits_for(fields.aligned_inputs)
        )
        student_logits = _single_sample(student_logits, name="student_logits")
        reference_logits = _single_sample(reference_logits, name="reference_logits")
        token_js = per_token_js_divergence(
            student_logits,
            reference_logits,
            vocabulary_support=self.config.sampling.vocabulary_support,
            top_k=self.config.sampling.top_k,
        )
        attention_mask = _single_sequence_mask(fields.aligned_inputs.attention_mask)
        completion_mask = _single_sequence_mask(fields.completion_mask)
        special_mask = (
            _single_sequence_mask(fields.special_position_mask)
            if fields.special_position_mask is not None
            else None
        )
        decision_mask = (
            _single_sequence_mask(fields.decision_mask)
            if fields.decision_mask is not None
            else None
        )
        valid_mask = build_valid_token_mask(
            attention_mask,
            token_scope=self.config.sampling.token_scope,
            completion_mask=completion_mask,
            decision_mask=decision_mask,
            special_position_mask=special_mask,
        )
        if valid_mask.shape != token_js.shape:
            raise ValueError("valid-token mask must align with causal logit positions")
        sample = sample_token_positions(
            token_js,
            valid_mask,
            self.config.sampling,
            global_step=global_step,
            sample_id=fields.sample_id,
            rank=rank,
        )
        student_states = {
            index: _single_sample(value, name=f"student layer {index}")
            for index, value in student_hidden_states.items()
        }
        anchor_states = {
            index: _single_sample(value, name=f"anchor layer {index}")
            for index, value in anchor_output.hidden_states.items()
        }
        loss = multi_layer_gram_loss(
            student_states,
            anchor_states,
            selection=self.selection,
            sampled_positions=sample.selected_positions,
            sampled_js_values=sample.selected_js_values,
            loss_weight=self.config.loss_weight,
            pair_weighting=self.config.pair_weighting,
            normalize_hidden_states=self.config.normalize_hidden_states,
            minimum_weight=self.config.sampling.minimum_weight,
        )
        component_masks = {
            name: [bool(value) for value in _single_sequence_mask(mask).tolist()]
            for name, mask in (fields.decision_component_masks or {}).items()
        }
        metrics = build_gram_anchor_metrics(
            loss,
            sample,
            self.selection,
            anchor_output.identity,
            global_step=global_step,
            global_loss_weight=self.config.loss_weight,
            js_sampling_mix=self.config.sampling.js_sampling_mix,
            decision_masks=component_masks,
        )
        return PrimeGramRegularizationStep(loss=loss, sample=sample, metrics=metrics)


def compose_training_objective(
    prime_policy_loss: Any,
    *,
    sdpo_loss: Any | None = None,
    gram_anchor_loss: Any | None = None,
) -> Any:
    """Compose independent policy, SDPO, and Gram losses.

    Args:
        prime_policy_loss: Required base Prime policy loss tensor/scalar.
        sdpo_loss: Optional already-weighted SDPO loss.
        gram_anchor_loss: Optional already-weighted Gram loss.

    Returns:
        ``prime_policy_loss`` plus each supplied auxiliary loss, preserving autograd.

    Example:
        ``total = compose_training_objective(policy, sdpo_loss=sdpo, gram_anchor_loss=gram)``

    Configuration remains separate: this function does not read ``ComponentWeights``
    or apply another Gram coefficient.
    """
    total = prime_policy_loss
    if sdpo_loss is not None:
        total = total + sdpo_loss
    if gram_anchor_loss is not None:
        total = total + gram_anchor_loss
    return total


def _single_sample(value: Any, *, name: str) -> Any:
    """Accept unbatched tensors or remove one singleton batch dimension."""
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} adapter path currently requires one sample")
        return value[0]
    if value.ndim != 2:
        raise ValueError(f"{name} must have [sequence, width] shape")
    return value


def _single_sequence_mask(value: Any) -> Any:
    """Accept a one-dimensional mask or remove one singleton batch dimension."""
    torch = __import__("torch")
    mask = torch.as_tensor(value)
    if mask.ndim == 2:
        if mask.shape[0] != 1:
            raise ValueError("Gram adapter currently requires one sample per mask")
        mask = mask[0]
    if mask.ndim != 1:
        raise ValueError("Gram masks must have a sequence dimension")
    return mask


__all__ = [
    "GramAnchorPrimeAdapter",
    "PrimeGramAnchorFields",
    "PrimeGramRegularizationStep",
    "SelectedLayerHookCapture",
    "compose_training_objective",
]
