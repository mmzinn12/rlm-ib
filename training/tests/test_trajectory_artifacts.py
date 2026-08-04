"""Verify versioned trajectory persistence, replay, masking, and rejudging.

Purpose:
    Ensure completed rollouts can be stored and recompiled without rerunning the
    student, while privileged judge content remains outside the artifact.
Implementation:
    A two-question trajectory is round-tripped through JSONL, tokenized with a
    deterministic character tokenizer, recompiled, and passed to a replacement judge.
Inputs:
    Synthetic trajectory/feedback values and temporary artifact storage.
Outputs:
    Assertions over persistence, masks, provenance, and replay summaries.
Example:
    Run ``pytest training/tests/test_trajectory_artifacts.py`` from the repository root.
"""

import json

import pytest
from rlm.core.trajectory import (
    CallItemSpan,
    DecisionKind,
    DecisionSpan,
    InvocationKind,
    InvocationNode,
    TrajectoryTree,
)

from rlm_train.judge.base import TaskContext
from rlm_train.judge.context import PrivilegedJudgeContext
from rlm_train.judge.schema import InformationValueFeedback, TrajectoryFeedback
from rlm_train.sdpo.masks import TokenOffset
from rlm_train.trajectory.artifacts import JSONLTrajectoryStore, TrajectoryArtifact
from rlm_train.trajectory.replay import (
    OfflineTrajectoryReplay,
    TokenizedContinuation,
    main,
)


class CharacterReplayTokenizer:
    """Map every response character to one deterministic test token."""

    fingerprint = "tokenizer-v1"

    def encode_with_offsets(self, continuation: str) -> TokenizedContinuation:
        """Return one token and one half-open offset per character."""
        return TokenizedContinuation(
            token_ids=tuple(range(len(continuation))),
            offsets=tuple(TokenOffset(index, index + 1) for index in range(len(continuation))),
        )


class CapturingJudge:
    """Capture the reconstructed task supplied during offline rejudging."""

    def __init__(self, feedback: TrajectoryFeedback) -> None:
        self.feedback = feedback
        self.tasks: list[TaskContext] = []

    async def evaluate(
        self,
        trajectory: TrajectoryTree,
        task: TaskContext,
    ) -> TrajectoryFeedback:
        """Return fixed feedback after retaining the replayed task."""
        trajectory.validate()
        self.tasks.append(task)
        return self.feedback


def make_trajectory_and_feedback() -> tuple[TrajectoryTree, TrajectoryFeedback]:
    """Build a complete two-question rollout with exact runtime bindings."""
    response = 'questions = ["first?", "second?"]'
    root_id = "run/root/i000"
    first_id = f"{root_id}/c000/b000"
    second_id = f"{root_id}/c000/b001"
    first_start = response.index('"first?"')
    second_start = response.index('"second?"')
    tree = TrajectoryTree(
        trajectory_id="run",
        nodes=[
            InvocationNode(
                node_id=root_id,
                parent_id=None,
                depth=0,
                kind=InvocationKind.ROOT,
                model="student",
                context="student-visible context",
                response=response,
                spans=[
                    DecisionSpan(
                        DecisionKind.CALL,
                        0,
                        len(response),
                        metadata={
                            "question_item_count": 2,
                            "addressable_question_item_count": 2,
                            "unaddressable_question_item_count": 0,
                        },
                    )
                ],
                call_item_spans=[
                    CallItemSpan(0, 0, first_start, first_start + len('"first?"'), first_id),
                    CallItemSpan(
                        0,
                        1,
                        second_start,
                        second_start + len('"second?"'),
                        second_id,
                    ),
                ],
            ),
            InvocationNode(
                node_id=first_id,
                parent_id=root_id,
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="student",
                context="first?",
                response="first answer",
                call_order=0,
                batch_index=0,
            ),
            InvocationNode(
                node_id=second_id,
                parent_id=root_id,
                depth=1,
                kind=InvocationKind.SUBCALL,
                model="student",
                context="second?",
                response="second answer",
                call_order=0,
                batch_index=1,
            ),
        ],
    )
    feedback = TrajectoryFeedback(
        trajectory_score=0.4,
        subcalls=[
            InformationValueFeedback(
                parent_node_id=root_id,
                child_node_id=first_id,
                information_significance=0.6,
                novelty=0.7,
                uncertainty_reduction=0.5,
                evidence_quality=0.8,
            ),
            InformationValueFeedback(
                parent_node_id=root_id,
                child_node_id=second_id,
                information_significance=0.1,
                novelty=0.2,
                uncertainty_reduction=0.1,
                evidence_quality=0.5,
            ),
        ],
        judge_version="judge-v1",
        rubric_version="rubric-v1",
    )
    return tree, feedback


def make_artifact(secret: str) -> tuple[TrajectoryArtifact, PrivilegedJudgeContext]:
    """Create a fully identified artifact and its external privileged payload."""
    tree, feedback = make_trajectory_and_feedback()
    privileged = PrivilegedJudgeContext("reference", "v1", {"answer": secret})
    task = TaskContext(
        "task-1",
        "public task prompt",
        evidence_snapshot={"public": "evidence"},
        privileged_context=privileged,
    )
    artifact = TrajectoryArtifact.from_task(
        artifact_id="artifact-1",
        task=task,
        dataset_id="dataset",
        dataset_revision="revision-1",
        example_id="example-7",
        context_references=("context://7",),
        student_model="student-model",
        student_checkpoint="checkpoint-1",
        policy_version=3,
        tokenizer_fingerprint="tokenizer-v1",
        trajectory=tree,
        feedback=feedback,
        trainer_configuration={"objective": "sdpo"},
        inference_configuration={"temperature": 0.0},
        experiment_configuration={"schema_version": 1, "resolved": True},
        feedback_projector={
            "name": "edge_local_question_feedback",
            "version": "v1",
            "mode": "diagnostic",
        },
        sampling_seeds={"rollout": 17},
        anchor_identity={"identifier": "anchor-1", "version": 0},
        teacher_identity={"identifier": "ema", "version": 4},
    )
    return artifact, privileged


def test_jsonl_artifact_round_trip_excludes_privileged_payload(tmp_path):
    secret = "never-persist-this-reference"
    artifact, _ = make_artifact(secret)
    store = JSONLTrajectoryStore(tmp_path / "rollouts.jsonl")

    store.append(artifact)
    loaded = list(store.iter_artifacts())

    assert len(loaded) == 1
    assert loaded[0].to_dict() == artifact.to_dict()
    assert loaded[0].privileged_context == artifact.privileged_context
    assert secret not in store.path.read_text()
    with pytest.raises(ValueError, match="duplicate"):
        store.append(artifact)


def test_offline_replay_rebuilds_isolated_question_masks(tmp_path, capsys):
    artifact, _ = make_artifact("external-only")
    store = JSONLTrajectoryStore(tmp_path / "rollouts.jsonl")
    store.append(artifact)

    result = OfflineTrajectoryReplay().compile_artifact(
        artifact,
        tokenizer=CharacterReplayTokenizer(),
    )

    assert len(result.node_examples) == 1
    assert len(result.question_examples) == 2
    assert len(result.tokenized_question_examples) == 2
    assert result.question_metrics.to_dict() == {
        "question_item_count": 2,
        "bound_question_item_count": 2,
        "unaddressable_question_item_count": 0,
        "question_feedback_count": 2,
    }
    first, second = result.tokenized_question_examples
    assert set(index for index, active in enumerate(first.question_mask) if active) == set(
        range(first.example.question_span.start, first.example.question_span.end)
    )
    assert not any(
        left and right
        for left, right in zip(first.question_mask, second.question_mask, strict=True)
    )
    with pytest.raises(ValueError, match="fingerprint"):
        OfflineTrajectoryReplay().compile_artifact(
            artifact,
            tokenizer=type(
                "WrongTokenizer",
                (),
                {
                    "fingerprint": "wrong",
                    "encode_with_offsets": CharacterReplayTokenizer().encode_with_offsets,
                },
            )(),
        )

    assert main([str(store.path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["artifact_id"] == artifact.artifact_id
    assert summary["question_example_count"] == 2


@pytest.mark.asyncio
async def test_offline_rejudge_requires_matching_privileged_provenance():
    artifact, privileged = make_artifact("expected-reference")
    _, replacement_feedback = make_trajectory_and_feedback()
    judge = CapturingJudge(replacement_feedback)
    replay = OfflineTrajectoryReplay()

    with pytest.raises(ValueError, match="does not match"):
        await replay.rejudge_artifact(
            artifact,
            judge,
            privileged_context=PrivilegedJudgeContext("reference", "v1", {"answer": "wrong"}),
        )

    updated = await replay.rejudge_artifact(
        artifact,
        judge,
        privileged_context=privileged,
    )

    assert updated.feedback == replacement_feedback
    assert judge.tasks[0].judge_payload()["privileged_context"]["payload"]["answer"] == (
        "expected-reference"
    )
