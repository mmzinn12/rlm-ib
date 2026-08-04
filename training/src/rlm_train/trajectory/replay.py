"""Compile stored trajectories offline without invoking the student policy again.

Purpose:
    Support judge replacement, deterministic recompilation, exact re-tokenization, and
    mask rebuilding from persisted rollout artifacts.
Implementation:
    ``OfflineTrajectoryReplay`` compiles node and isolated-question examples from one
    artifact. An optional tokenizer protocol produces exact continuation offsets used
    to rebuild component and question masks. Rejudging requires the original privileged
    context fingerprint when one was recorded.
Inputs:
    Versioned trajectory artifacts, optional replacement feedback, an optional exact
    tokenizer adapter, and an optional structured judge.
Outputs:
    Replay compilations containing examples, token IDs, masks, and coverage metrics.
Example:
    ``result = OfflineTrajectoryReplay().compile_artifact(artifact)``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from rlm.core.trajectory import DecisionKind

from rlm_train.judge.base import TaskContext, TrajectoryJudge
from rlm_train.judge.context import PrivilegedJudgeContext
from rlm_train.judge.schema import TrajectoryFeedback
from rlm_train.sdpo.masks import (
    TokenOffset,
    build_exclusive_token_masks,
    build_question_token_mask,
)
from rlm_train.trajectory.artifacts import JSONLTrajectoryStore, TrajectoryArtifact
from rlm_train.trajectory.compiler import (
    NodeTrainingExample,
    QuestionTrainingExample,
    TrajectoryCompiler,
)
from rlm_train.trajectory.validation import QuestionTraceMetrics, summarize_question_trace


@dataclass(frozen=True)
class TokenizedContinuation:
    """Store exact continuation token IDs and response-relative character offsets."""

    token_ids: tuple[int, ...]
    offsets: tuple[TokenOffset, ...]

    def __post_init__(self) -> None:
        """Require one valid offset per non-negative token ID."""
        if len(self.token_ids) != len(self.offsets):
            raise ValueError("token IDs and offsets must have identical length")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token IDs must be non-negative")


class ReplayTokenizer(Protocol):
    """Adapt the exact student tokenizer to offline continuation replay."""

    @property
    def fingerprint(self) -> str:
        """Return the stable tokenizer fingerprint recorded in artifacts."""
        ...

    def encode_with_offsets(self, continuation: str) -> TokenizedContinuation:
        """Tokenize text and return response-relative offsets for every token."""
        ...


@dataclass(frozen=True)
class TokenizedNodeTrainingExample:
    """Attach replayed tokens and exclusive component masks to one node example."""

    example: NodeTrainingExample
    continuation: TokenizedContinuation
    component_masks: dict[DecisionKind, tuple[bool, ...]]


@dataclass(frozen=True)
class TokenizedQuestionTrainingExample:
    """Attach replayed tokens and one isolated question mask to an edge example."""

    example: QuestionTrainingExample
    continuation: TokenizedContinuation
    question_mask: tuple[bool, ...]


@dataclass(frozen=True)
class ReplayCompilation:
    """Return framework-neutral examples and optional exact tokenization products."""

    artifact_id: str
    node_examples: tuple[NodeTrainingExample, ...]
    question_examples: tuple[QuestionTrainingExample, ...]
    tokenized_node_examples: tuple[TokenizedNodeTrainingExample, ...]
    tokenized_question_examples: tuple[TokenizedQuestionTrainingExample, ...]
    question_metrics: QuestionTraceMetrics

    def to_summary(self) -> dict[str, int | str]:
        """Return a compact JSON-compatible replay summary."""
        return {
            "artifact_id": self.artifact_id,
            "node_example_count": len(self.node_examples),
            "question_example_count": len(self.question_examples),
            "tokenized_node_example_count": len(self.tokenized_node_examples),
            "tokenized_question_example_count": len(self.tokenized_question_examples),
            **self.question_metrics.to_dict(),
        }


class OfflineTrajectoryReplay:
    """Rejudge and recompile stored rollout artifacts deterministically."""

    def __init__(self, compiler: TrajectoryCompiler | None = None) -> None:
        """Use the supplied compiler or create the standard trajectory compiler."""
        self.compiler = compiler or TrajectoryCompiler()

    def compile_artifact(
        self,
        artifact: TrajectoryArtifact,
        *,
        feedback: TrajectoryFeedback | None = None,
        tokenizer: ReplayTokenizer | None = None,
        on_unaddressable: Literal["error", "skip"] = "error",
    ) -> ReplayCompilation:
        """Compile examples and optionally rebuild exact token masks offline."""
        resolved_feedback = feedback or artifact.feedback
        if resolved_feedback is None:
            raise ValueError("offline compilation requires stored or replacement feedback")
        node_examples = tuple(self.compiler.compile(artifact.trajectory, resolved_feedback))
        question_examples = tuple(
            self.compiler.compile_questions(
                artifact.trajectory,
                resolved_feedback,
                on_unaddressable=on_unaddressable,
            )
        )
        metrics = summarize_question_trace(
            artifact.trajectory,
            question_feedback_count=len(resolved_feedback.subcalls),
        )
        tokenized_nodes: tuple[TokenizedNodeTrainingExample, ...] = ()
        tokenized_questions: tuple[TokenizedQuestionTrainingExample, ...] = ()
        if tokenizer is not None:
            if tokenizer.fingerprint != artifact.tokenizer_fingerprint:
                raise ValueError("replay tokenizer fingerprint does not match artifact")
            tokenized_by_node: dict[str, TokenizedContinuation] = {}

            def tokenize(node_id: str, continuation: str) -> TokenizedContinuation:
                cached = tokenized_by_node.get(node_id)
                if cached is not None:
                    return cached
                encoded = tokenizer.encode_with_offsets(continuation)
                if any(offset.end > len(continuation) for offset in encoded.offsets):
                    raise ValueError("tokenizer offset exceeds replayed continuation")
                tokenized_by_node[node_id] = encoded
                return encoded

            node_results: list[TokenizedNodeTrainingExample] = []
            for example in node_examples:
                encoded = tokenize(example.node_id, example.continuation)
                node_results.append(
                    TokenizedNodeTrainingExample(
                        example=example,
                        continuation=encoded,
                        component_masks={
                            kind: tuple(mask)
                            for kind, mask in build_exclusive_token_masks(
                                example.spans,
                                list(encoded.offsets),
                            ).items()
                        },
                    )
                )
            tokenized_nodes = tuple(node_results)
            question_results: list[TokenizedQuestionTrainingExample] = []
            for example in question_examples:
                encoded = tokenize(example.parent_node_id, example.student_continuation)
                mask = tuple(
                    build_question_token_mask(example.question_span, list(encoded.offsets))
                )
                if not any(mask):
                    raise ValueError("question span does not overlap any replayed token")
                question_results.append(
                    TokenizedQuestionTrainingExample(
                        example=example,
                        continuation=encoded,
                        question_mask=mask,
                    )
                )
            tokenized_questions = tuple(question_results)
        return ReplayCompilation(
            artifact_id=artifact.artifact_id,
            node_examples=node_examples,
            question_examples=question_examples,
            tokenized_node_examples=tokenized_nodes,
            tokenized_question_examples=tokenized_questions,
            question_metrics=metrics,
        )

    async def rejudge_artifact(
        self,
        artifact: TrajectoryArtifact,
        judge: TrajectoryJudge,
        *,
        privileged_context: PrivilegedJudgeContext | None = None,
    ) -> TrajectoryArtifact:
        """Replace judge feedback without rerunning the stored student trajectory."""
        supplied_descriptor = privileged_context.descriptor() if privileged_context else None
        if artifact.privileged_context is not None:
            if supplied_descriptor is None:
                raise ValueError("rejudging requires the recorded privileged context")
            if supplied_descriptor != artifact.privileged_context:
                raise ValueError("privileged context does not match artifact provenance")
        task = TaskContext(
            task_id=artifact.task_id,
            prompt=artifact.task_prompt,
            evidence_snapshot=artifact.task_evidence_snapshot,
            metadata=dict(artifact.task_metadata),
            privileged_context=privileged_context,
        )
        feedback = await judge.evaluate(artifact.trajectory, task)
        return replace(
            artifact,
            feedback=feedback,
            privileged_context=supplied_descriptor or artifact.privileged_context,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Compile stored artifacts and emit one JSON summary per input record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_path", help="Path to a versioned trajectory JSONL file")
    parser.add_argument(
        "--on-unaddressable",
        choices=("error", "skip"),
        default="error",
        help="How to handle judged questions without exact source spans",
    )
    args = parser.parse_args(argv)
    replay = OfflineTrajectoryReplay()
    store = JSONLTrajectoryStore(args.artifact_path)
    for artifact in store.iter_artifacts():
        result = replay.compile_artifact(
            artifact,
            on_unaddressable=args.on_unaddressable,
        )
        print(json.dumps(result.to_summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OfflineTrajectoryReplay",
    "ReplayCompilation",
    "ReplayTokenizer",
    "TokenizedContinuation",
    "TokenizedNodeTrainingExample",
    "TokenizedQuestionTrainingExample",
    "main",
]
