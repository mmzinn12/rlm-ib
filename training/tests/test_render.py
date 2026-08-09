"""Readable recursion-tree rendering over canonical annotated rollouts."""

from __future__ import annotations

from rlm_train.trajectory.render import render_recursion_tree
from rlm_train.trajectory.schema import (
    AnnotatedRollout,
    ExecutionEdge,
    ExecutionNode,
    ExecutionRecord,
    FeedbackRecord,
    NodeRole,
    TaskPartition,
)


def nested_rollout() -> AnnotatedRollout:
    categorical_assessment = {
        "assessment_id": "assess-1",
        "focal_edge_ids": ["edge-root-child"],
        "focal_node_ids": ["child"],
        "provider": "openai:categorical",
        "scope": "retrospective_local",
        "content": {
            "judge_mode": "categorical",
            "categories": {
                "significance": "medium",
                "novelty": "high",
                "uncertainty_reduction": "low",
                "evidence_quality": "good",
            },
            "information_significance": 0.6,
            "novelty": 1.0,
            "uncertainty_reduction": 0.25,
            "evidence_quality": 1.0,
            "redundant": False,
            "misleading": False,
            "diagnostic": "asked a focused decomposition question",
            "information_revealed": ["capital city"],
        },
    }
    return AnnotatedRollout(
        rollout_id="rollout-render",
        mode="training",
        task=TaskPartition(task_id="task", public={"prompt": "answer the dense question"}),
        policy={"policy_owner": "student"},
        execution=ExecutionRecord(
            root_node_id="root",
            nodes=(
                ExecutionNode(
                    node_id="root",
                    role=NodeRole.ROOT,
                    depth=0,
                    prompt="answer the dense question",
                    result="FINAL",
                ),
                ExecutionNode(
                    node_id="child",
                    parent_id="root",
                    role=NodeRole.RECURSIVE_SUBCALL,
                    depth=1,
                    prompt="what is the capital?",
                    result="the capital is X",
                ),
                ExecutionNode(
                    node_id="grandchild",
                    parent_id="child",
                    role=NodeRole.PLAIN_SUBCALL,
                    depth=2,
                    prompt="which country?",
                    result="country Y",
                ),
            ),
            edges=(
                ExecutionEdge(
                    edge_id="edge-root-child",
                    parent_id="root",
                    child_id="child",
                    kind="recursive",
                    question="what is the capital?",
                ),
                ExecutionEdge(
                    edge_id="edge-child-grandchild",
                    parent_id="child",
                    child_id="grandchild",
                    kind="plain",
                    question="which country?",
                ),
            ),
            events=({"sequence_number": 0}, {"sequence_number": 1}),
        ),
        feedback=FeedbackRecord(judge_assessments=(categorical_assessment,)),
        result={"final_answer": "FINAL"},
    )


def test_render_recursion_tree_shows_questions_answers_and_scores():
    rendered = render_recursion_tree(nested_rollout())

    assert "Rollout rollout-render  (mode=training)" in rendered
    assert "Final answer: FINAL" in rendered
    assert "Q: what is the capital?" in rendered
    assert "A: the capital is X" in rendered
    assert "judge[categorical] via openai:categorical" in rendered
    assert "significance=medium(0.60)" in rendered
    assert "asked a focused decomposition question" in rendered
    # Nested plain subcall is rendered beneath its recursive parent.
    assert "Q: which country?" in rendered
    assert "A: country Y" in rendered


def test_render_recursion_tree_uses_tree_connectors_and_indentation():
    rendered = render_recursion_tree(nested_rollout())
    lines = rendered.splitlines()

    connectors = ("\u2514\u2500 ", "\u251c\u2500 ")
    assert any(line.startswith(connectors) for line in lines)
    # The grandchild question is indented under the child branch.
    grandchild_line = next(line for line in lines if "Q: which country?" in line)
    assert grandchild_line.startswith("   ")
