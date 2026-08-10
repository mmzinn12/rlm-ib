"""SDPO loss and full provider+trainer stack validated on CPU with a fake policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlm_train.artifacts.rollout_json import RolloutJSONWriter
from rlm_train.datasets.records import DatasetRecord
from rlm_train.engine.providers import (
    JudgeFeedbackProvider,
    SelfDistillationTeacherTargetProvider,
    TransformersPolicyScoreProvider,
)
from rlm_train.engine.trainer import CanonicalTrainer
from rlm_train.feedback.schema import FeedbackBundle
from rlm_train.judge.providers.fake import DeterministicFakeJudge
from rlm_train.metrics import MetricCollector
from rlm_train.models.identity import PolicyIdentity, TokenizerIdentity
from rlm_train.models.protocol import PolicyScore
from rlm_train.objectives.build import build_objective_composer
from rlm_train.objectives.protocol import ObjectiveBatch
from rlm_train.objectives.sdpo.loss import build_sdpo_compute_loss
from rlm_train.objectives.sdpo.target_support import extract_topk_teacher_target
from rlm_train.rollouts.protocol import RolloutResult
from rlm_train.rollouts.selectors import select_tokens
from rlm_train.spec.feedback import AssessmentScope
from rlm_train.spec.objectives import ObjectivesSpec, SDPOSpec, TokenScope
from rlm_train.spec.run import DatasetRefSpec, RunSpec, RuntimeSpec
from rlm_train.spec.artifacts import ArtifactSpec
from rlm_train.spec.models import StudentSpec
from rlm_train.teachers.targets import TeacherTarget
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    AnnotationRecord,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    GenerationTokens,
    NodeRole,
    TaskPartition,
)

VOCAB = 8
TOP_K = 4
FEEDBACK_MARKER = 4242


def test_sdpo_compute_loss_is_positive_and_differentiable_when_teacher_differs():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    student_bias = torch.nn.Parameter(torch.randn(VOCAB))
    student_logits = student_bias.unsqueeze(0).expand(3, VOCAB)
    teacher = extract_topk_teacher_target(
        torch.randn(3, VOCAB),
        top_k=TOP_K,
        teacher_version=0,
        tokenizer_fingerprint="fp",
    )
    target = TeacherTarget(
        target_id="t",
        rollout_id="r",
        generation_id="g",
        selected_token_ids=(0, 1, 2),
        selected_positions=(0, 1, 2),
        topk_token_ids=teacher.token_ids,
        topk_logprobs=teacher.logprobs,
        tail_logprob_mass=teacher.tail_logprobs,
        teacher_fingerprint="f",
        tokenizer_fingerprint="fp",
        configuration_fingerprint="c",
    )
    batch = ObjectiveBatch(
        rollouts=(SimpleNamespace(rollout_id="r"),),
        token_selections={},
        policy_scores={"r": student_logits},
        teacher_targets={"r": target},
        feedback=FeedbackBundle(),
    )

    result = build_sdpo_compute_loss(SDPOSpec(enabled=True, weight=1.0, top_k=TOP_K))(batch)

    assert result.active_token_count == 3
    assert result.loss.item() > 0.0
    result.loss.backward()
    assert student_bias.grad is not None
    assert torch.isfinite(student_bias.grad).all()


class FakePolicy:
    def __init__(self, vocab: int) -> None:
        torch = __import__("torch")
        self.bias = torch.nn.Parameter(torch.randn(vocab))
        self._identity = PolicyIdentity(
            component_id="student",
            revision="default",
            policy_owner="student",
            checkpoint_id="latest",
        )
        self._tokenizer = TokenizerIdentity(
            component_id="student", revision="default", vocabulary_size=vocab
        )

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def tokenizer_identity(self) -> TokenizerIdentity:
        return self._tokenizer

    def generate(self, request):  # pragma: no cover - unused in scoring tests
        raise NotImplementedError

    def score_sampled_ids(self, generation, *, require_grad, return_logits=False, **_):
        torch = __import__("torch")
        count = len(generation.token_ids)
        logits = self.bias.unsqueeze(0).expand(count, self.bias.shape[0])
        if FEEDBACK_MARKER in generation.prompt_token_ids:
            shift = torch.zeros(self.bias.shape[0])
            shift[0] = 6.0
            logits = logits + shift
        if not require_grad:
            logits = logits.detach()
        targets = torch.tensor(generation.token_ids, dtype=torch.long)
        logprobs = torch.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
        return PolicyScore(
            token_ids=generation.token_ids,
            logprobs=logprobs,
            logits=logits if return_logits else None,
        )

    def tokenize(self, text):
        return (FEEDBACK_MARKER,)

    def trainable_parameters(self):
        return (self.bias,)


def training_rollout() -> AnnotatedRollout:
    generation = GenerationTokens(
        generation_id="g-root",
        node_id="root",
        policy_owner="student",
        text="abcd",
        prompt_token_ids=(9,),
        token_ids=(0, 1, 2, 3),
        token_offsets=((0, 1), (1, 2), (2, 3), (3, 4)),
    )
    return AnnotatedRollout(
        rollout_id="train-rollout",
        mode="training",
        task=TaskPartition(task_id="task", public={"prompt": "task"}),
        policy={"policy_owner": "student"},
        execution=ExecutionRecord(
            root_node_id="root",
            nodes=(
                ExecutionNode(
                    node_id="root", role=NodeRole.ROOT, depth=0, prompt="task", result="answer"
                ),
                ExecutionNode(
                    node_id="child",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    prompt="ask",
                    result="sub-answer",
                ),
            ),
            edges=(
                ExecutionEdge(
                    edge_id="edge-1",
                    parent_id="root",
                    child_id="child",
                    kind="plain",
                    question="ask",
                ),
            ),
            events=(
                {
                    "event_type": "helper_question_generated",
                    "event_id": "e0",
                    "sequence_number": 0,
                    "invocation_id": "root",
                    "subcall_id": "edge-1",
                    "question": "ask",
                },
            ),
        ),
        annotations=AnnotationRecord(generations=(generation,)),
        result={"final_answer": "answer"},
    )


def test_full_provider_stack_runs_one_optimizer_step(tmp_path):
    torch = pytest.importorskip("torch")
    rollout = training_rollout()
    policy = FakePolicy(VOCAB)

    class Dataset:
        def records(self):
            return (DatasetRecord(record_id="task", public_task={"prompt": "task"}),)

    class Engine:
        def execute(self, request):
            del request
            return RolloutResult(completion=None, rollout=rollout)

    spec = RunSpec(
        student=StudentSpec(model_id="student", policy_owner="student"),
        training_dataset=DatasetRefSpec(source="unused.jsonl"),
        objectives=ObjectivesSpec(
            sdpo=SDPOSpec(
                enabled=True, weight=1.0, token_scope=TokenScope.ALL_STUDENT_TOKENS, top_k=TOP_K
            )
        ),
        artifacts=ArtifactSpec(output_directory=str(tmp_path)),
        runtime=RuntimeSpec(max_optimizer_steps=1, learning_rate=0.1),
    )
    metrics = MetricCollector()
    trainer = CanonicalTrainer(
        spec=spec,
        dataset=Dataset(),
        rollout_engine=Engine(),
        objectives=build_objective_composer(spec.objectives),
        optimizer=torch.optim.AdamW(policy.trainable_parameters(), lr=spec.runtime.learning_rate),
        policy_scores=TransformersPolicyScoreProvider(policy),
        policy_owner="student",
        policy_parameters=tuple(policy.trainable_parameters()),
        feedback=JudgeFeedbackProvider(DeterministicFakeJudge()),
        teacher_targets=SelfDistillationTeacherTargetProvider(policy, top_k=TOP_K),
        artifact_writer=RolloutJSONWriter(tmp_path / "rollouts"),
        metric_recorder=metrics,
    )

    result = trainer.train()

    assert result.state.optimizer_step == 1
    assert len(result.rollout_artifacts) == 1
    assert {item.name for item in metrics.observations} == {
        "train/loss/total",
        "train/optimizer/gradient_norm",
    }
    saved = AnnotatedRollout.model_validate_json(
        (tmp_path / "rollouts" / "train-rollout.json").read_text()
    )
    assert saved.teacher_targets
    assert saved.feedback.judge_assessments


def test_rubric_feedback_conditions_teacher_and_makes_loss_positive():
    from rlm_train.objectives.protocol import ObjectiveCapabilities

    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    rollout = training_rollout()
    policy = FakePolicy(VOCAB)
    selections = {
        rollout.rollout_id: select_tokens(
            rollout,
            objective="sdpo",
            token_scope=TokenScope.ALL_STUDENT_TOKENS,
            policy_owner="student",
        )
    }
    capabilities = ObjectiveCapabilities(token_scope=TokenScope.ALL_STUDENT_TOKENS)
    student = TransformersPolicyScoreProvider(policy).score(
        "sdpo", capabilities, (rollout,), selections
    )
    teacher_provider = SelfDistillationTeacherTargetProvider(policy, top_k=TOP_K)

    rubric = {
        "information_revealed": ["the release year"],
        "what_was_missing": "the specific film title",
        "redundant_with_context": False,
        "misleading_or_invalid": False,
        "why_it_mattered": "it would disambiguate the two candidates",
        "improved_question_guidance": "name the film and ask for its director",
        "rationale": "the question was too broad",
    }
    scopes = frozenset({AssessmentScope.RETROSPECTIVE_LOCAL})
    with_feedback = JudgeFeedbackProvider(
        DeterministicFakeJudge(content={"rubric": rubric})
    ).assess(object(), (rollout,), scopes)
    without_feedback = FeedbackBundle()

    compute_loss = build_sdpo_compute_loss(SDPOSpec(enabled=True, weight=1.0, top_k=TOP_K))

    def loss_for(feedback: FeedbackBundle) -> float:
        targets = teacher_provider.build("sdpo", (rollout,), selections, feedback)
        batch = ObjectiveBatch(
            rollouts=(rollout,),
            token_selections=selections,
            policy_scores=student.policy_scores,
            teacher_targets=targets,
            feedback=feedback,
        )
        return compute_loss(batch).loss.item()

    assert loss_for(with_feedback) > 0.0
    assert loss_for(without_feedback) == pytest.approx(0.0, abs=1e-6)


def test_build_objective_composer_requires_enabled_objective():
    with pytest.raises(ValueError, match="at least one objective"):
        build_objective_composer(ObjectivesSpec())


def make_bare_trainer(policy_owner: str) -> CanonicalTrainer:
    # Only policy_owner is consulted by _select_objective_tokens; other collaborators are unused.
    spec = RunSpec(student=StudentSpec(model_id="student", policy_owner=policy_owner))
    return CanonicalTrainer(
        spec=spec,
        dataset=object(),
        rollout_engine=object(),
        objectives=object(),
        optimizer=object(),
        policy_scores=object(),
        policy_owner=policy_owner,
        policy_parameters=(),
    )


def test_select_objective_tokens_returns_none_when_no_tokens_selected():
    from rlm_train.objectives.protocol import ObjectiveCapabilities

    trainer = make_bare_trainer("student")
    capabilities = {"sdpo": ObjectiveCapabilities(token_scope=TokenScope.ALL_STUDENT_TOKENS)}
    rollout = training_rollout()

    # Student-owned generation -> selectable.
    assert trainer._select_objective_tokens((rollout,), capabilities) is not None

    # Same rollout but the generation is owned by someone else -> nothing selected -> None.
    other_generation = rollout.annotations.generations[0].model_copy(
        update={"policy_owner": "other"}
    )
    other_annotations = rollout.annotations.model_copy(
        update={"generations": (other_generation,)}
    )
    other_rollout = rollout.model_copy(update={"annotations": other_annotations})
    assert trainer._select_objective_tokens((other_rollout,), capabilities) is None
