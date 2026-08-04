"""Verify locked SDPO settings, reverse KL, masks, and Prime payload invariants.

Purpose:
    Protect the agreed reverse-KL/EMA/top-k/depth-1 configuration and its mathematical
    and transport boundaries.
Implementation:
    Deterministic tests compare the production reverse KL with SciPy, verify gradient
    boundaries, exercise mask precedence, validate normalized teacher targets, and
    check tokenizer identity.
Inputs:
    Small probability vectors, spans, token offsets, and Pydantic configuration values.
Outputs:
    Pytest assertions and expected validation exceptions.
Example:
    Run ``pytest training/tests/test_sdpo.py`` from the repository root.
"""

import math

import pytest
from pydantic import ValidationError
from rlm.core.trajectory import CallItemSpan, DecisionKind, DecisionSpan
from scipy.special import rel_entr

from rlm_train.sdpo.config import ComponentWeights, SDPOConfig
from rlm_train.sdpo.loss import reverse_kl_topk_with_tail
from rlm_train.sdpo.masks import (
    TokenOffset,
    build_exclusive_token_masks,
    build_question_token_mask,
)
from rlm_train.sdpo.prime_adapter import PrimeQuestionSDPOFields, PrimeTreeSDPOFields
from rlm_train.sdpo.teacher import TopKTeacherTarget


def test_mvp_config_is_reverse_kl_fixed_topk_depth_one():
    config = SDPOConfig()

    assert config.divergence == "reverse_kl"
    assert config.teacher == "fixed"
    assert config.ema_update_rate is None
    assert config.top_k == 100
    assert config.include_tail_bucket is True
    assert config.max_depth == 1
    assert config.mask_overlap == "exclusive"
    assert config.allow_privileged_evidence is False


def test_config_rejects_an_inactive_objective():
    with pytest.raises(ValidationError, match="at least one"):
        SDPOConfig(
            component_weights=ComponentWeights(
                route=0,
                call=0,
                node=0,
                aggregation=0,
                final=0,
                missing_call=0,
            )
        )


def test_reverse_kl_is_zero_for_identical_coarsened_distributions():
    torch = pytest.importorskip("torch")
    topk = torch.tensor([[math.log(0.5), math.log(0.3)]])
    tail = torch.tensor([math.log(0.2)])

    loss = reverse_kl_topk_with_tail(topk, tail, topk, tail, torch.tensor([True]))

    assert loss.item() == pytest.approx(0.0)


def test_reverse_kl_uses_student_as_the_left_distribution():
    torch = pytest.importorskip("torch")
    student = [0.6, 0.3]
    student_tail = 0.1
    teacher = [0.4, 0.4]
    teacher_tail = 0.2
    student_logprobs = torch.tensor(
        [[math.log(value) for value in student]],
        requires_grad=True,
    )
    student_tail_logprobs = torch.tensor([math.log(student_tail)], requires_grad=True)
    teacher_logprobs = torch.tensor(
        [[math.log(value) for value in teacher]],
        requires_grad=True,
    )
    teacher_tail_logprobs = torch.tensor([math.log(teacher_tail)], requires_grad=True)

    loss = reverse_kl_topk_with_tail(
        student_logprobs,
        student_tail_logprobs,
        teacher_logprobs,
        teacher_tail_logprobs,
        torch.tensor([True]),
    )
    expected = float(sum(rel_entr([*student, student_tail], [*teacher, teacher_tail])))
    loss.backward()

    assert loss.item() == pytest.approx(expected)
    assert student_logprobs.grad is not None
    assert student_tail_logprobs.grad is not None
    assert teacher_logprobs.grad is None
    assert teacher_tail_logprobs.grad is None


def test_token_masks_use_exclusive_precedence():
    spans = [
        DecisionSpan(kind=DecisionKind.ROUTE, start=0, end=8),
        DecisionSpan(kind=DecisionKind.CALL, start=4, end=12),
    ]
    offsets = [TokenOffset(0, 4), TokenOffset(4, 8), TokenOffset(8, 12)]

    masks = build_exclusive_token_masks(spans, offsets)

    assert masks[DecisionKind.ROUTE] == [True, False, False]
    assert masks[DecisionKind.CALL] == [False, True, True]


def test_question_mask_excludes_sibling_questions_and_list_punctuation():
    span = CallItemSpan(call_order=0, batch_index=1, start=8, end=16)
    offsets = [
        TokenOffset(0, 1),
        TokenOffset(1, 7),
        TokenOffset(7, 8),
        TokenOffset(8, 12),
        TokenOffset(12, 16),
        TokenOffset(16, 17),
    ]

    assert build_question_token_mask(span, offsets) == [
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_teacher_target_requires_aligned_topk_rows():
    with pytest.raises(ValidationError, match="token IDs and logprobs"):
        TopKTeacherTarget(
            token_ids=[[1, 2]],
            logprobs=[[-0.1]],
            tail_logprobs=[-2.0],
            teacher_version=0,
            tokenizer_fingerprint="tokenizer-v1",
        )


def test_prime_payload_enforces_same_tokenizer_and_exclusive_masks():
    target = TopKTeacherTarget(
        token_ids=[[1, 2], [1, 2]],
        logprobs=[
            [math.log(0.5), math.log(0.3)],
            [math.log(0.6), math.log(0.2)],
        ],
        tail_logprobs=[math.log(0.2), math.log(0.2)],
        teacher_version=0,
        tokenizer_fingerprint="tokenizer-v1",
    )
    payload = PrimeTreeSDPOFields(
        trajectory_id="run",
        node_id="root",
        component_masks={
            DecisionKind.CALL: [True, False],
            DecisionKind.FINAL: [False, True],
        },
        teacher_target=target,
    )

    payload.validate(token_count=2, student_tokenizer_fingerprint="tokenizer-v1")
    with pytest.raises(ValueError, match="fingerprints"):
        payload.validate(token_count=2, student_tokenizer_fingerprint="other")


def test_question_payload_requires_one_non_empty_isolated_mask():
    target = TopKTeacherTarget(
        token_ids=[[1], [1]],
        logprobs=[[math.log(0.8)], [math.log(0.8)]],
        tail_logprobs=[math.log(0.2), math.log(0.2)],
        teacher_version=0,
        tokenizer_fingerprint="tokenizer-v1",
    )
    payload = PrimeQuestionSDPOFields(
        trajectory_id="run",
        parent_node_id="root",
        child_node_id="child",
        call_order=0,
        batch_index=1,
        question_mask=[False, True],
        teacher_target=target,
    )

    payload.validate(token_count=2, student_tokenizer_fingerprint="tokenizer-v1")
