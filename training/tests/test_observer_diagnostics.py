"""Verify epistemic, reasoning, topology, and divergence observations are pure."""

import copy

import pytest
from rlm.core.trajectory import CallItemSpan, InvocationKind, InvocationNode, TrajectoryTree

from rlm_train.diagnostics import (
    build_gram_observer_metrics,
    collect_observer_diagnostics,
    effective_rank_from_singular_values,
)


def make_tree() -> TrajectoryTree:
    response = 'questions = ["q?"]'
    root_id = "run/root/i000"
    child_id = f"{root_id}/c000/b000"
    start = response.index('"q?"')
    return TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id=root_id,
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="model",
                context="prompt",
                response=response,
                call_item_spans=[CallItemSpan(0, 0, start, start + 4, child_id)],
                metadata={"retry_count": 1},
            ),
            InvocationNode(
                node_id=child_id,
                parent_id=root_id,
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="model",
                context="q?",
                call_order=0,
                batch_index=0,
            ),
        ],
    )


def test_observer_diagnostics_count_markers_revisions_and_trajectory_without_mutation():
    response = "Wait, maybe another approach. Check the result; actually, I was wrong."
    tokens = tuple(response.split())
    tree = make_tree()
    before = copy.deepcopy(tree.to_dict())

    observed = collect_observer_diagnostics(
        response,
        response_tokens=tokens,
        truncated=True,
        trajectory=tree,
        per_token_divergence=tuple(0.1 * (index + 1) for index in range(len(tokens))),
        gram_metrics={"gram/drift": 0.2},
    )

    assert observed.epistemic.counts["wait"] == 1
    assert observed.epistemic.counts["maybe"] == 1
    assert observed.reasoning.abandoned_approach_count == 1
    assert observed.reasoning.derived_result_check_count >= 1
    assert observed.reasoning.correction_count >= 2
    assert observed.trajectory.question_count == 1
    assert observed.trajectory.subcall_count == 1
    assert observed.trajectory.maximum_depth == 1
    assert observed.trajectory.retry_count == 1
    assert observed.divergence.epistemic_position_count >= 2
    assert observed.gram == {"gram/drift": 0.2}
    assert tree.to_dict() == before


def test_divergence_requires_exact_token_alignment():
    with pytest.raises(ValueError, match="align"):
        collect_observer_diagnostics(
            "wait now",
            response_tokens=("wait", "now"),
            per_token_divergence=(0.1,),
        )


def test_gram_observer_records_per_layer_loss_and_effective_rank():
    metrics = build_gram_observer_metrics(
        gram_drift=0.25,
        per_layer_gram_losses={3: 0.1, 7: 0.2},
        per_layer_singular_values={3: (1.0, 1.0), 7: (3.0, 0.0)},
    )

    assert metrics["per_layer_gram_loss"] == {3: 0.1, 7: 0.2}
    assert metrics["per_layer_effective_rank"][3] == pytest.approx(2.0)
    assert metrics["per_layer_effective_rank"][7] == pytest.approx(1.0)
    assert effective_rank_from_singular_values((0.0, 0.0)) == 0.0
