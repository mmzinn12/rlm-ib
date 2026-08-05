"""Contract tests for RunSpec, structural selectors, and judge evidence boundaries."""

import json

import pytest
from pydantic import ValidationError

from rlm_train.artifacts import RolloutJSONWriter
from rlm_train.engine import CanonicalTrainer, PolicyScoreBatch
from rlm_train.feedback import FeedbackVisibility
from rlm_train.judge.views import build_judge_view
from rlm_train.metrics import MetricCollector
from rlm_train.objectives import (
    ObjectiveCapabilities,
    ObjectiveComposer,
    ObjectiveResult,
)
from rlm_train.rollouts.protocol import RolloutResult
from rlm_train.rollouts.selectors import select_tokens
from rlm_train.rollouts.semantics import annotate_generation
from rlm_train.spec import (
    AssessmentScope,
    DatasetRefSpec,
    ObjectivesSpec,
    RunSpec,
    RuntimeSpec,
    SDPOSpec,
    StudentSpec,
    TokenScope,
)
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    AnnotationRecord,
    DecisionRole,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    GenerationTokens,
    NodeRole,
    TaskPartition,
)


def character_generation(
    generation_id: str, node_id: str, text: str, owner: str = "student"
) -> GenerationTokens:
    return GenerationTokens(
        generation_id=generation_id,
        node_id=node_id,
        policy_owner=owner,
        text=text,
        prompt_token_ids=(9001,),
        token_ids=tuple(range(len(text))),
        token_offsets=tuple((index, index + 1) for index in range(len(text))),
    )


def representative_rollout() -> AnnotatedRollout:
    root_text = "reasoning\n```repl\nvalue = llm_query('useful question')\n```"
    root_generation = character_generation("g-root", "root", root_text)
    child_generation = character_generation("g-child", "child-1", "direct response")
    spans = (
        *annotate_generation(
            root_generation,
            node_role=NodeRole.ROOT,
            default_decision_role=DecisionRole.REASONING,
        ),
        *annotate_generation(
            child_generation,
            node_role=NodeRole.PLAIN_SUBCALL,
            default_decision_role=DecisionRole.SUBCALL_RESPONSE,
        ),
    )
    events = (
        {
            "event_type": "invocation_started",
            "event_id": "event-0",
            "sequence_number": 0,
            "invocation_id": "root",
            "prompt": "task",
        },
        {
            "event_type": "helper_question_generated",
            "event_id": "event-1",
            "sequence_number": 1,
            "invocation_id": "root",
            "subcall_id": "edge-1",
            "question": "useful question",
        },
        {
            "event_type": "subcall_completed",
            "event_id": "event-2",
            "sequence_number": 2,
            "invocation_id": "root",
            "subcall_id": "edge-1",
            "response": "direct response",
        },
        {
            "event_type": "helper_question_generated",
            "event_id": "event-3",
            "sequence_number": 3,
            "invocation_id": "root",
            "subcall_id": "edge-2",
            "question": "sibling question",
        },
        {
            "event_type": "subcall_completed",
            "event_id": "event-4",
            "sequence_number": 4,
            "invocation_id": "root",
            "subcall_id": "edge-2",
            "response": "SIBLING_SECRET",
        },
        {
            "event_type": "final_answer_submitted",
            "event_id": "event-5",
            "sequence_number": 5,
            "invocation_id": "root",
            "answer": "FINAL_SENTINEL",
        },
    )
    return AnnotatedRollout(
        rollout_id="rollout",
        mode="training",
        task=TaskPartition(task_id="task", public={"prompt": "task"}),
        policy={"policy_owner": "student"},
        execution=ExecutionRecord(
            root_node_id="root",
            nodes=(
                ExecutionNode(
                    node_id="root",
                    role=NodeRole.ROOT,
                    depth=0,
                    policy_owner="student",
                    prompt="task",
                    result="FINAL_SENTINEL",
                ),
                ExecutionNode(
                    node_id="child-1",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    policy_owner="student",
                    prompt="useful question",
                    result="direct response",
                ),
                ExecutionNode(
                    node_id="child-2",
                    parent_id="root",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=1,
                    policy_owner="student",
                    prompt="sibling question",
                    result="SIBLING_SECRET",
                ),
            ),
            edges=(
                ExecutionEdge(
                    edge_id="edge-1",
                    parent_id="root",
                    child_id="child-1",
                    kind="plain",
                    question="useful question",
                ),
                ExecutionEdge(
                    edge_id="edge-2",
                    parent_id="root",
                    child_id="child-2",
                    kind="plain",
                    question="sibling question",
                ),
            ),
            events=events,
        ),
        annotations=AnnotationRecord(
            generations=(root_generation, child_generation), semantic_spans=spans
        ),
        result={"final_answer": "FINAL_SENTINEL"},
    )


def test_run_spec_accepts_canonical_full_rlm_shape_and_is_immutable():
    spec = RunSpec(
        student=StudentSpec(model_id="student"),
        training_dataset=DatasetRefSpec(source="train.jsonl"),
        objectives=ObjectivesSpec(
            sdpo=SDPOSpec(enabled=True, weight=1.0, token_scope=TokenScope.HELPER_QUESTIONS)
        ),
    )

    assert spec.rollout.engine == "rlm"
    assert spec.evaluation.recursive_policy is True
    assert RunSpec.model_validate_json(spec.canonical_json()) == spec
    with pytest.raises(ValidationError):
        spec.rollout.max_depth = 1


def test_each_objective_scope_selects_structural_student_owned_ranges():
    rollout = representative_rollout()
    results = {
        scope: select_tokens(
            rollout,
            objective=scope.value,
            token_scope=scope,
            policy_owner="student",
        )
        for scope in TokenScope
    }
    root = rollout.annotations.generations[0]
    helper_positions = [
        index
        for index, active in enumerate(results[TokenScope.HELPER_QUESTIONS].masks["g-root"])
        if active
    ]

    assert "".join(root.text[index] for index in helper_positions) == "useful question"
    assert results[TokenScope.SUBCALL_NATURAL_LANGUAGE].durable.active_token_count == len(
        "direct response"
    )
    assert results[TokenScope.ALL_STUDENT_TOKENS].durable.active_token_count == sum(
        len(generation.token_ids) for generation in rollout.annotations.generations
    )
    natural_positions = results[TokenScope.NATURAL_LANGUAGE].masks["g-root"]
    selected_text = "".join(
        root.text[index] for index, active in enumerate(natural_positions) if active
    )
    assert "reasoning" in selected_text
    assert "useful question" in selected_text
    assert "llm_query" not in selected_text


def test_local_judge_views_exclude_siblings_final_answer_and_private_reference():
    rollout = representative_rollout()
    retrospective = build_judge_view(
        rollout,
        scope=AssessmentScope.RETROSPECTIVE_LOCAL,
        focal_edge_ids=("edge-1",),
        allowed_objectives=frozenset({"sdpo"}),
        allowed_token_scopes=frozenset({TokenScope.HELPER_QUESTIONS}),
    )
    causal = build_judge_view(
        rollout,
        scope=AssessmentScope.CAUSAL_LOCAL,
        focal_edge_ids=("edge-1",),
        verifier_reference="PRIVATE_SECRET",
    )
    retrospective_json = json.dumps(retrospective.model_dump(mode="json"), sort_keys=True)
    causal_json = json.dumps(causal.model_dump(mode="json"), sort_keys=True)

    assert retrospective.visibility is FeedbackVisibility.RESTRICTED
    assert "direct response" in retrospective_json
    assert "SIBLING_SECRET" not in retrospective_json
    assert "FINAL_SENTINEL" not in retrospective_json
    assert "PRIVATE_SECRET" not in retrospective_json
    assert "direct response" not in causal_json
    assert "FINAL_SENTINEL" not in causal_json
    assert "PRIVATE_SECRET" not in causal_json


def test_non_privileged_view_is_invariant_to_final_result_replacement():
    rollout = representative_rollout()
    first = build_judge_view(
        rollout,
        scope=AssessmentScope.CAUSAL_LOCAL,
        focal_edge_ids=("edge-1",),
    )
    changed_root = rollout.execution.nodes[0].model_copy(update={"result": "CHANGED"})
    changed_execution = rollout.execution.model_copy(
        update={"nodes": (changed_root, *rollout.execution.nodes[1:])}
    )
    changed = rollout.model_copy(
        update={
            "execution": changed_execution,
            "result": {"final_answer": "CHANGED"},
        }
    )
    second = build_judge_view(
        changed,
        scope=AssessmentScope.CAUSAL_LOCAL,
        focal_edge_ids=("edge-1",),
    )

    assert first.fingerprint == second.fingerprint


def test_canonical_trainer_performs_real_capability_driven_optimizer_step(tmp_path):
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    rollout = representative_rollout().model_copy(update={"rollout_id": "train-rollout"})

    class Dataset:
        def records(self):
            from rlm_train.datasets import DatasetRecord

            return (DatasetRecord(record_id="task", public_task={"prompt": "task"}),)

    class Engine:
        def execute(self, request):
            del request
            return RolloutResult(completion=None, rollout=rollout)

    class Scores:
        def score(self, objective, capabilities, rollouts, selections):
            del objective, capabilities, rollouts, selections
            return PolicyScoreBatch(policy_scores={"parameter": parameter})

    class QuadraticObjective:
        @property
        def capabilities(self):
            return ObjectiveCapabilities(token_scope=TokenScope.ALL_STUDENT_TOKENS)

        def compute(self, batch):
            loss = batch.policy_scores["parameter"].square()
            return ObjectiveResult(
                loss=loss,
                active_token_count=sum(
                    selection.active_token_count for selection in batch.token_selections.values()
                ),
            )

    spec = RunSpec(
        student=StudentSpec(model_id="student", policy_owner="student"),
        runtime=RuntimeSpec(max_optimizer_steps=1, learning_rate=0.1),
    )
    metrics = MetricCollector()
    trainer = CanonicalTrainer(
        spec=spec,
        dataset=Dataset(),
        rollout_engine=Engine(),
        objectives=ObjectiveComposer({"quadratic": (1.0, QuadraticObjective())}),
        optimizer=torch.optim.SGD((parameter,), lr=spec.runtime.learning_rate),
        policy_scores=Scores(),
        policy_owner="student",
        policy_parameters=(parameter,),
        artifact_writer=RolloutJSONWriter(tmp_path / "rollouts"),
        metric_recorder=metrics,
    )

    result = trainer.train()

    assert result.state.optimizer_step == 1
    assert parameter.item() < 1.0
    assert len(result.rollout_artifacts) == 1
    assert {item.name for item in metrics.observations} == {
        "train/loss/total",
        "train/optimizer/gradient_norm",
    }
    saved = AnnotatedRollout.model_validate_json(
        (tmp_path / "rollouts" / "train-rollout.json").read_text()
    )
    assert saved.annotations.objective_selections["quadratic"].active_token_count > 0
