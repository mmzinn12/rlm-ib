"""Test JS scoring, masking, sampling, Gram math, gradients, and anchor lifecycle.

Purpose:
    Protect the tensor-level mathematical and autograd invariants of Gram anchoring.
Implementation:
    Small deterministic PyTorch examples compare production tensors with SciPy,
    replay sampling, inspect gradients, rotate features, and exercise periodic anchor
    refresh behavior.
Inputs:
    Synthetic logits, hidden states, masks, JS values, and configuration models.
Outputs:
    Pytest assertions over divergence, sampling, loss, gradient, and lifecycle results.
Example:
    Run ``pytest training/tests/test_regularization.py`` from the repository root.
"""

import math

import pytest
from scipy.spatial.distance import jensenshannon

from rlm_train.regularization.anchor import (
    AlignedSequenceInputs,
    PeriodicEMASnapshotAnchorController,
)
from rlm_train.regularization.config import GramLayerSelectionConfig, JSTokenSamplingConfig
from rlm_train.regularization.divergence import (
    coarsen_logits_to_reference_topk,
    per_token_js_divergence,
)
from rlm_train.regularization.gram import gram_matrix_loss, multi_layer_gram_loss
from rlm_train.regularization.metrics import summarize_js
from rlm_train.regularization.sampling import sample_token_positions
from rlm_train.regularization.selectors import (
    build_completion_mask,
    build_valid_token_mask,
    resolve_layer_selection,
)

torch = pytest.importorskip("torch")


def test_per_token_js_is_zero_for_identical_logits():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.3, 0.4]])

    divergence = per_token_js_divergence(logits, logits, top_k=2)

    assert torch.allclose(divergence, torch.zeros_like(divergence), atol=1e-7)


def test_full_support_per_token_js_matches_scipy_and_is_symmetric():
    student_probabilities = [0.8, 0.2]
    reference_probabilities = [0.1, 0.9]
    student_logits = torch.tensor([student_probabilities]).log()
    reference_logits = torch.tensor([reference_probabilities]).log()

    forward = per_token_js_divergence(
        student_logits,
        reference_logits,
        vocabulary_support="full",
    )
    reverse = per_token_js_divergence(
        reference_logits,
        student_logits,
        vocabulary_support="full",
    )
    expected = jensenshannon(student_probabilities, reference_probabilities) ** 2

    assert forward.item() == pytest.approx(expected, abs=1e-7)
    assert forward.item() == pytest.approx(reverse.item(), abs=1e-7)
    assert 0.0 <= forward.item() <= math.log(2.0)


def test_reference_topk_plus_tail_preserves_probability_mass():
    student = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    reference = torch.tensor([[0.0, 3.0, 1.0, -2.0]])

    result = coarsen_logits_to_reference_topk(student, reference, top_k=2)
    divergence = per_token_js_divergence(student, reference, top_k=2)
    expected = (
        jensenshannon(
            result.student_probabilities.squeeze(0).numpy(),
            result.reference_probabilities.squeeze(0).numpy(),
        )
        ** 2
    )

    assert result.student_probabilities.sum(dim=-1).item() == pytest.approx(1.0)
    assert result.reference_probabilities.sum(dim=-1).item() == pytest.approx(1.0)
    assert result.student_probabilities.shape[-1] == 3
    assert divergence.item() == pytest.approx(expected, abs=1e-7)


def test_js_summary_uses_numpy_linear_quantiles():
    summary = summarize_js([0.0, 1.0, 2.0, 3.0])
    singleton = summarize_js([0.4])

    assert summary.mean == pytest.approx(1.5)
    assert summary.maximum == pytest.approx(3.0)
    assert summary.q50 == pytest.approx(1.5)
    assert summary.q90 == pytest.approx(2.7)
    assert summary.q99 == pytest.approx(2.97)
    assert singleton.q50 == singleton.q90 == singleton.q99 == pytest.approx(0.4)


def test_completion_scope_excludes_prompt_padding_and_special_positions():
    attention = torch.tensor([0, 1, 1, 1, 1, 0])
    special = torch.tensor([0, 0, 0, 0, 1, 0])
    completion = build_completion_mask(
        attention, completion_start_positions=3, special_position_mask=special
    )

    valid = build_valid_token_mask(
        attention,
        token_scope="completion",
        completion_mask=completion,
        special_position_mask=special,
    )

    assert valid.tolist() == [False, False, False, True, False, False]


def test_zero_js_sampling_is_uniform_unique_and_uses_all_short_sequences():
    config = JSTokenSamplingConfig(sample_size=8, seed=7)
    selection = sample_token_positions(
        torch.zeros(4),
        torch.tensor([1, 1, 0, 1]),
        config,
        global_step=2,
        sample_id="sample",
    )

    assert selection.selected_positions == (0, 1, 3)
    assert len(set(selection.selected_positions)) == selection.sampled_token_count
    assert selection.valid_probabilities == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_high_js_position_is_selected_more_frequently_across_seeded_steps():
    config = JSTokenSamplingConfig(
        sample_size=1,
        js_sampling_mix=1.0,
        minimum_weight=1e-8,
        seed=17,
    )
    counts = [0, 0, 0]
    for step in range(120):
        selection = sample_token_positions(
            torch.tensor([0.01, 0.01, 1.0]),
            torch.ones(3),
            config,
            global_step=step,
            sample_id="sample",
        )
        counts[selection.selected_positions[0]] += 1

    assert counts[2] > counts[0] + counts[1]


def test_gram_zero_non_negative_and_shared_rotation_invariant():
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    anchor = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

    zero = gram_matrix_loss(student, student, pair_weighting="none").weighted
    original = gram_matrix_loss(student, anchor, pair_weighting="none").weighted
    rotated = gram_matrix_loss(
        student @ rotation, anchor @ rotation, pair_weighting="none"
    ).weighted

    assert zero.item() == pytest.approx(0.0)
    assert original.item() >= 0.0
    assert rotated.item() == pytest.approx(original.item())


def test_weighted_gram_matches_scalar_reference_and_detaches_anchor_and_js():
    student = torch.tensor([[1.0, 0.0], [1.0, 1.0]], requires_grad=True)
    anchor = torch.tensor([[0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    js = torch.tensor([0.2, 0.8], requires_grad=True)

    result = gram_matrix_loss(student, anchor, token_weights=js)
    normalized_student = torch.nn.functional.normalize(student.float(), dim=-1)
    normalized_anchor = torch.nn.functional.normalize(anchor.detach().float(), dim=-1)
    squared = (
        normalized_student @ normalized_student.T - normalized_anchor @ normalized_anchor.T
    ).square()
    weights = js.detach() / js.detach().mean()
    expected = (weights[:, None] * weights[None, :] * squared).sum() / (
        weights[:, None] * weights[None, :]
    ).sum()
    result.weighted.backward()

    assert result.weighted.item() == pytest.approx(expected.item())
    assert student.grad is not None
    assert anchor.grad is None
    assert js.grad is None


def test_multi_layer_weights_normalize_then_global_weight_scales_and_zero_disables():
    selection = resolve_layer_selection(
        GramLayerSelectionConfig(indices=(0, 1), layer_weights=(1.0, 3.0)),
        block_count=2,
    )
    student = {
        0: torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        1: torch.tensor([[1.0, 1.0], [1.0, 0.0]], requires_grad=True),
    }
    anchor = {
        0: torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        1: torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    }

    result = multi_layer_gram_loss(
        student,
        anchor,
        selection=selection,
        sampled_positions=(0, 1),
        sampled_js_values=(0.2, 0.8),
        loss_weight=2.0,
    )
    expected = 2.0 * (result.layer_losses[0] + 3.0 * result.layer_losses[1]) / 4.0
    disabled = multi_layer_gram_loss(
        student,
        anchor,
        selection=selection,
        sampled_positions=(0, 1),
        sampled_js_values=(0.2, 0.8),
        loss_weight=0.0,
    )

    assert result.total_loss.item() == pytest.approx(expected.item())
    assert result.effective_layer_weights == pytest.approx({0: 0.5, 1: 1.5})
    assert disabled.total_loss.item() == pytest.approx(0.0)
    assert result.sampled_positions == disabled.sampled_positions == (0, 1)


def test_periodic_anchor_inputs_have_no_feedback_channel_and_refresh_on_interval():
    inputs = AlignedSequenceInputs(input_ids=torch.tensor([1, 2]), attention_mask=torch.ones(2))

    def forward_model(model, aligned_inputs, layer_indices):
        del aligned_inputs
        logits = model.clone().requires_grad_(True)
        states = {index: model.clone().requires_grad_(True) for index in layer_indices}
        return logits, states

    controller = PeriodicEMASnapshotAnchorController(
        3,
        snapshot_model=lambda model: model.detach().clone(),
        forward_model=forward_model,
    )

    assert "feedback" not in inputs.__dataclass_fields__
    assert controller.maybe_refresh(torch.ones(2, 3), global_step=0) is True
    assert controller.maybe_refresh(torch.zeros(2, 3), global_step=2) is False
    assert controller.maybe_refresh(torch.zeros(2, 3), global_step=3) is True
    output = controller.forward(inputs, (0,))
    assert output.identity.version == 1
    assert output.logits.requires_grad is False
    assert output.hidden_states[0].requires_grad is False
