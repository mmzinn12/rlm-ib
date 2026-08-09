"""Numerical and schema tests for the canonical SDPO package."""

import math

import pytest
from pydantic import ValidationError
from scipy.special import rel_entr

from rlm_train.objectives.sdpo import TopKTeacherTarget, reverse_kl_topk_with_tail
from rlm_train.objectives.sdpo.target_support import extract_topk_teacher_target


def test_reverse_kl_is_zero_for_identical_distributions():
    torch = pytest.importorskip("torch")
    topk = torch.tensor([[math.log(0.5), math.log(0.3)]])
    tail = torch.tensor([math.log(0.2)])

    loss = reverse_kl_topk_with_tail(topk, tail, topk, tail, torch.tensor([True]))

    assert loss.item() == pytest.approx(0.0)


def test_reverse_kl_uses_student_as_left_distribution_and_detaches_teacher():
    torch = pytest.importorskip("torch")
    student = [0.6, 0.3]
    teacher = [0.4, 0.4]
    student_logprobs = torch.tensor([[math.log(value) for value in student]], requires_grad=True)
    student_tail = torch.tensor([math.log(0.1)], requires_grad=True)
    teacher_logprobs = torch.tensor([[math.log(value) for value in teacher]], requires_grad=True)
    teacher_tail = torch.tensor([math.log(0.2)], requires_grad=True)

    loss = reverse_kl_topk_with_tail(
        student_logprobs,
        student_tail,
        teacher_logprobs,
        teacher_tail,
        torch.tensor([True]),
    )
    expected = float(sum(rel_entr([*student, 0.1], [*teacher, 0.2])))
    loss.backward()

    assert loss.item() == pytest.approx(expected)
    assert student_logprobs.grad is not None
    assert student_tail.grad is not None
    assert teacher_logprobs.grad is None
    assert teacher_tail.grad is None


def test_teacher_target_requires_aligned_topk_rows():
    with pytest.raises(ValidationError, match="token IDs and logprobs"):
        TopKTeacherTarget(
            token_ids=((1, 2),),
            logprobs=((-0.1,),),
            tail_logprobs=(-2.0,),
            teacher_version=0,
            tokenizer_fingerprint="tokenizer-v1",
        )


def test_topk_target_extraction_retains_tail_probability_mass():
    torch = pytest.importorskip("torch")
    target = extract_topk_teacher_target(
        torch.tensor([[2.0, 1.0, 0.0]]),
        top_k=2,
        teacher_version=3,
        tokenizer_fingerprint="tokenizer-v1",
    )
    retained_mass = sum(math.exp(value) for value in target.logprobs[0])

    assert retained_mass + math.exp(target.tail_logprobs[0]) == pytest.approx(1.0)
    assert target.teacher_version == 3


def test_topk_target_extraction_normalizes_large_vocab_within_tolerance():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    target = extract_topk_teacher_target(
        torch.randn(8, 151_936, dtype=torch.float32),
        top_k=100,
        teacher_version=0,
        tokenizer_fingerprint="normalization-test",
    )
    masses = [
        sum(math.exp(value) for value in row) + math.exp(tail)
        for row, tail in zip(target.logprobs, target.tail_logprobs, strict=True)
    ]

    assert max(abs(mass - 1.0) for mass in masses) < 1e-8
