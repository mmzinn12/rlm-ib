"""Feedback-free fixed-anchor Gram loss for exact sampled continuations."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from rlm_train.colab.generation import model_device
from rlm_train.colab.objectives import TrainingBatch
from rlm_train.regularization import (
    GramAnchorConfig,
    multi_layer_gram_loss,
    per_token_js_divergence,
    resolve_layer_selection,
    sample_token_positions,
)
from rlm_train.sdpo import model_state_fingerprint


class TransformersGramLossBuilder:
    """Recompute aligned hidden states only when the Gram term is enabled."""

    def __init__(
        self,
        *,
        student: Any,
        anchor: Any,
        configuration: GramAnchorConfig,
        global_step: Callable[[], int] = lambda: 0,
    ) -> None:
        torch = _torch()
        if not configuration.is_active:
            raise ValueError("Gram loss builder requires an active Gram configuration")
        if not isinstance(student, torch.nn.Module) or not isinstance(anchor, torch.nn.Module):
            raise TypeError("student and anchor must be PyTorch modules")
        self.student = student
        self.anchor = anchor
        self.configuration = configuration
        self.global_step = global_step
        self.anchor.eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        block_count = int(getattr(student.config, "num_hidden_layers", 0))
        if block_count <= 0:
            raise ValueError("student config must declare num_hidden_layers")
        anchor_block_count = int(getattr(anchor.config, "num_hidden_layers", 0))
        if anchor_block_count != block_count:
            raise ValueError("student and anchor transformer block counts differ")
        self.selection = resolve_layer_selection(configuration.layers, block_count=block_count)
        self.anchor_fingerprint = model_state_fingerprint(anchor)

    @classmethod
    def from_student(
        cls,
        student: Any,
        *,
        configuration: GramAnchorConfig,
        global_step: Callable[[], int] = lambda: 0,
    ) -> TransformersGramLossBuilder:
        """Capture the initialized pre-training policy as an independent frozen anchor."""
        return cls(
            student=student,
            anchor=copy.deepcopy(student),
            configuration=configuration,
            global_step=global_step,
        )

    @property
    def identity(self) -> dict[str, Any]:
        """Return fixed anchor source, fingerprint, and resolved layers."""
        return {
            "strategy": self.configuration.anchor.strategy,
            "checkpoint_path": self.configuration.anchor.checkpoint_path,
            "fingerprint": self.anchor_fingerprint,
            "resolved_layers": list(self.selection.indices),
        }

    async def prepare(self, batch: TrainingBatch) -> TrainingBatch:
        """Perform no model work; aligned anchor work belongs to the loss factory."""
        batch.validate()
        return batch

    def loss(
        self,
        batch: TrainingBatch,
        continuation_logits: Mapping[str, Any],
    ) -> tuple[Any, int]:
        """Compute detached-JS-sampled multi-layer geometry loss per rollout."""
        torch = _torch()
        sample_losses: list[Any] = []
        sample_weights: list[int] = []
        student_device = model_device(self.student)
        anchor_device = model_device(self.anchor)
        for sample in batch.samples:
            complete = (*sample.prompt_token_ids, *sample.continuation_token_ids)
            student_ids = torch.tensor([complete], dtype=torch.long, device=student_device)
            student_attention = torch.ones_like(student_ids)
            student_output = self.student(
                input_ids=student_ids,
                attention_mask=student_attention,
                output_hidden_states=True,
            )
            anchor_ids = student_ids.to(anchor_device)
            anchor_attention = student_attention.to(anchor_device)
            with torch.inference_mode():
                anchor_output = self.anchor(
                    input_ids=anchor_ids,
                    attention_mask=anchor_attention,
                    output_hidden_states=True,
                )
            start = len(sample.prompt_token_ids) - 1
            stop = start + len(sample.continuation_token_ids)
            anchor_logits = anchor_output.logits[0, start:stop, :].float().to(student_device)
            student_logits = continuation_logits[sample.trajectory_id]
            if student_logits.shape != anchor_logits.shape:
                raise ValueError("student and anchor continuation logits are not aligned")
            js = per_token_js_divergence(
                student_logits,
                anchor_logits,
                vocabulary_support=self.configuration.sampling.vocabulary_support,
                top_k=self.configuration.sampling.top_k,
            )
            valid = sample.trainable_token_mask.to(device=js.device, dtype=torch.bool)
            selection = sample_token_positions(
                js,
                valid,
                self.configuration.sampling,
                global_step=int(self.global_step()),
                sample_id=sample.trajectory_id,
            )
            student_hidden = {
                layer_index: student_output.hidden_states[layer_index + 1][0, start:stop, :]
                for layer_index in self.selection.indices
            }
            anchor_hidden = {
                layer_index: anchor_output.hidden_states[layer_index + 1][0, start:stop, :].to(
                    student_device
                )
                for layer_index in self.selection.indices
            }
            result = multi_layer_gram_loss(
                student_hidden,
                anchor_hidden,
                selection=self.selection,
                sampled_positions=selection.selected_positions,
                sampled_js_values=selection.selected_js_values,
                loss_weight=1.0,
                pair_weighting=self.configuration.pair_weighting,
                normalize_hidden_states=self.configuration.normalize_hidden_states,
                minimum_weight=self.configuration.sampling.minimum_weight,
            )
            sample_losses.append(result.total_loss * selection.sampled_token_count)
            sample_weights.append(selection.sampled_token_count)
        active = sum(sample_weights)
        if active <= 0:
            raise ValueError("Gram objective contains no sampled continuation tokens")
        self.validate_anchor_unchanged()
        return torch.stack(sample_losses).sum() / active, active

    def validate_anchor_unchanged(self) -> None:
        """Fail if optimizer or model movement mutated the frozen anchor."""
        if model_state_fingerprint(self.anchor) != self.anchor_fingerprint:
            raise RuntimeError("fixed Gram anchor changed during training")

    def validate_unchanged(self) -> None:
        """Implement the trainer's immutable auxiliary-model guard."""
        self.validate_anchor_unchanged()


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Transformers Gram anchoring") from exc


__all__ = ["TransformersGramLossBuilder"]
