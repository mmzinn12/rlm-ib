"""Verify concrete EMA, question scoring, top-k/tail, caching, and aggregation.

Purpose:
    Protect the framework-neutral SDPO execution path before any Prime adapter owns it.
Implementation:
    Tiny PyTorch tensors and a fake question-logit provider exercise every mathematical
    and information-isolation boundary without loading a model or running a rollout.
Inputs:
    Deterministic logits, one isolated question example, masks, and a linear model.
Outputs:
    Assertions over EMA lifecycle, cache identity, probability mass, and gradients.
Example:
    Run ``pytest training/tests/test_sdpo_execution.py`` from the repository root.
"""

import json
from dataclasses import replace
from typing import Any

import pytest
from rlm.core.trajectory import CallItemSpan, DecisionKind

from rlm_train.judge.context import PrivilegedJudgeContext
from rlm_train.judge.schema import (
    DiagnosticQuestionTeacherFeedback,
    InformationValueFeedback,
    QuestionTeacherFeedback,
)
from rlm_train.sdpo.cache import MemoryTeacherTargetCache
from rlm_train.sdpo.config import ComponentWeights
from rlm_train.sdpo.loss import (
    gather_student_topk_with_tail,
    reverse_kl_topk_with_tail,
    teacher_target_tensors,
    weighted_component_reverse_kl,
)
from rlm_train.sdpo.teacher import (
    TopKQuestionTeacherScorer,
    TorchEMATeacherController,
    build_question_feedback_context,
    extract_topk_teacher_target,
)
from rlm_train.trajectory.compiler import QuestionTrainingExample


class FakeQuestionLogitsProvider:
    """Return fixed teacher logits and capture the scorer's restricted inputs."""

    def __init__(self, logits: Any, *, teacher_version: int = 0) -> None:
        self.logits = logits
        self.teacher_version = teacher_version
        self.calls: list[dict[str, Any]] = []

    async def score_existing_continuation(
        self,
        *,
        student_context: Any,
        student_continuation: str,
        feedback: QuestionTeacherFeedback,
    ) -> Any:
        """Return logits after retaining only the allowed question inputs."""
        self.calls.append(
            {
                "student_context": student_context,
                "student_continuation": student_continuation,
                "feedback": build_question_feedback_context(feedback),
            }
        )
        return self.logits


def make_question_example() -> QuestionTrainingExample:
    """Build one edge-isolated question example."""
    continuation = 'questions = ["active?"]'
    start = continuation.index('"active?"')
    return QuestionTrainingExample(
        trajectory_id="run",
        parent_node_id="run/root/i000",
        child_node_id="run/root/i000/c000/b000",
        call_order=0,
        batch_index=0,
        student_context={"prompt": "public student context"},
        student_continuation=continuation,
        question_span=CallItemSpan(
            call_order=0,
            batch_index=0,
            start=start,
            end=start + len('"active?"'),
            child_node_id="run/root/i000/c000/b000",
        ),
        feedback=DiagnosticQuestionTeacherFeedback(
            projector_version="v1",
            parent_node_id="run/root/i000",
            child_node_id="run/root/i000/c000/b000",
            information_significance=0.7,
            uncertainty_reduction=0.6,
            novelty=0.8,
            evidence_quality=0.9,
            diagnostic="The active question is useful but needs a narrower scope.",
        ),
    )


def test_ema_teacher_is_frozen_versioned_and_checkpointable():
    torch = pytest.importorskip("torch")
    student = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
    controller = TorchEMATeacherController.from_student(student, version=3)
    with torch.no_grad():
        student.weight.fill_(4.0)

    controller.update_after_optimizer_step(student, update_rate=0.25)

    assert controller.version == 4
    torch.testing.assert_close(controller.teacher.weight, torch.full((1, 2), 2.5))
    assert not controller.teacher.training
    assert all(not parameter.requires_grad for parameter in controller.teacher.parameters())
    restored = TorchEMATeacherController.from_student(student)
    restored.load_state_dict(controller.state_dict())
    assert restored.version == 4
    torch.testing.assert_close(restored.teacher.weight, torch.full((1, 2), 2.5))


def test_topk_tail_gather_preserves_only_student_gradients():
    torch = pytest.importorskip("torch")
    teacher_logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -2.0]],
        requires_grad=True,
    )
    target = extract_topk_teacher_target(
        teacher_logits,
        top_k=2,
        teacher_version=5,
        tokenizer_fingerprint="tokenizer-v1",
    )
    student_logits = torch.tensor(
        [[1.5, 0.5, 0.2, -0.3], [0.3, 1.2, 1.0, -0.5]],
        requires_grad=True,
    )
    student = gather_student_topk_with_tail(student_logits, target)
    teacher_topk, teacher_tail = teacher_target_tensors(target, reference=student.logprobs)

    torch.testing.assert_close(
        torch.exp(student.logprobs).sum(dim=-1) + torch.exp(student.tail_logprobs),
        torch.ones(2),
    )
    loss = reverse_kl_topk_with_tail(
        student.logprobs,
        student.tail_logprobs,
        teacher_topk,
        teacher_tail,
        torch.tensor([True, True]),
    )
    loss.backward()

    assert student_logits.grad is not None
    assert teacher_logits.grad is None
    assert not teacher_topk.requires_grad
    assert not teacher_tail.requires_grad


@pytest.mark.asyncio
async def test_question_scorer_uses_restricted_context_and_versioned_cache():
    torch = pytest.importorskip("torch")
    secret = "privileged reference must not cross this boundary"
    privileged = PrivilegedJudgeContext("answer-key", "v1", {"reference": secret})
    provider = FakeQuestionLogitsProvider(
        torch.tensor([[2.0, 1.0, 0.0]] * len(make_question_example().student_continuation)),
        teacher_version=7,
    )
    cache = MemoryTeacherTargetCache()
    scorer = TopKQuestionTeacherScorer(
        provider,
        top_k=2,
        tokenizer_fingerprint="tokenizer-v1",
        feedback_version="rubric-v1",
        cache=cache,
    )
    example = make_question_example()

    first = await scorer.score_question(example)
    second = await scorer.score_question(example)

    assert first == second
    assert len(provider.calls) == 1
    assert provider.calls[0]["feedback"]["child_node_id"] == example.child_node_id
    assert "trajectory_score" not in provider.calls[0]["feedback"]
    assert "rationale" not in provider.calls[0]["feedback"]
    assert secret not in json.dumps(provider.calls[0])
    assert secret not in repr(privileged.descriptor())
    complete_feedback = InformationValueFeedback(
        parent_node_id=example.parent_node_id,
        child_node_id=example.child_node_id,
        information_significance=0.5,
        novelty=0.5,
        uncertainty_reduction=0.5,
        evidence_quality=0.5,
        rationale="complete judge rationale",
    )
    with pytest.raises(TypeError, match="QuestionTeacherFeedback"):
        build_question_feedback_context(complete_feedback)
    with pytest.raises(TypeError, match="QuestionTeacherFeedback"):
        await scorer.score_question(
            replace(
                example,
                feedback=complete_feedback,
            )
        )

    provider.teacher_version = 8
    assert (await scorer.score_question(example)).teacher_version == 8
    assert len(provider.calls) == 2


def test_component_losses_are_token_normalized_then_weighted():
    torch = pytest.importorskip("torch")
    teacher_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    target = extract_topk_teacher_target(
        teacher_logits,
        top_k=2,
        teacher_version=0,
        tokenizer_fingerprint="tokenizer-v1",
    )
    student_logits = torch.tensor(
        [[1.0, 1.5, 0.0], [0.5, 0.0, 1.5]],
        requires_grad=True,
    )
    weights = ComponentWeights(
        route=0,
        call=2.0,
        node=0,
        aggregation=0,
        final=0.5,
        missing_call=0,
    )

    result = weighted_component_reverse_kl(
        student_logits,
        target,
        {
            DecisionKind.CALL: [True, False],
            DecisionKind.FINAL: [False, True],
        },
        weights,
    )

    assert result.total.item() == pytest.approx(
        2.0 * result.component_losses[DecisionKind.CALL].item()
        + 0.5 * result.component_losses[DecisionKind.FINAL].item()
    )
    assert result.active_token_counts == {
        DecisionKind.CALL: 1,
        DecisionKind.FINAL: 1,
    }
    result.total.backward()
    assert student_logits.grad is not None
    with pytest.raises(ValueError, match="exclusive"):
        weighted_component_reverse_kl(
            student_logits.detach(),
            target,
            {
                DecisionKind.CALL: [True, False],
                DecisionKind.FINAL: [True, False],
            },
            weights,
        )
