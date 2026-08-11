"""Concrete training providers: differentiable student scores, judge feedback, teacher targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from rlm_train.engine.trainer import PolicyScoreBatch
from rlm_train.feedback.schema import FeedbackBundle, ScopedAssessment
from rlm_train.judge.protocol import Judge
from rlm_train.judge.views import build_judge_view
from rlm_train.models.protocol import SampledGeneration, TrainablePolicy
from rlm_train.objectives.protocol import ObjectiveCapabilities
from rlm_train.objectives.sdpo.target_support import extract_topk_teacher_target
from rlm_train.rollouts.selectors import TokenSelectionResult
from rlm_train.spec.feedback import AssessmentScope
from rlm_train.teachers.targets import TeacherTarget
from rlm_train.trajectory.schema import AnnotatedRollout, ObjectiveSelection


def selected_generation_positions(
    selection: ObjectiveSelection,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Group every selected token position by generation in durable range order."""
    positions_by_generation: dict[str, list[int]] = {}
    for item in selection.ranges:
        positions_by_generation.setdefault(item.generation_id, []).extend(
            range(item.token_start, item.token_end)
        )
    groups = tuple(
        (generation_id, tuple(positions))
        for generation_id, positions in positions_by_generation.items()
        if positions
    )
    if not groups:
        raise ValueError("objective selection has no selected token positions")
    return groups


def reconstruct_generation(
    rollout: AnnotatedRollout, generation_id: str, policy: TrainablePolicy
) -> SampledGeneration:
    generation = next(
        item for item in rollout.annotations.generations if item.generation_id == generation_id
    )
    return SampledGeneration(
        text=generation.text,
        prompt_token_ids=generation.prompt_token_ids,
        token_ids=generation.token_ids,
        token_offsets=generation.token_offsets,
        policy=policy.identity,
        tokenizer=policy.tokenizer_identity,
    )


def generation_edge_ids(rollout: AnnotatedRollout, generation_id: str) -> frozenset[str]:
    """Return helper edges emitted while executing one student generation.

    Canonical events preserve the active generation and the later helper-question event in the
    same invocation. Older synthetic or persisted records without generation lifecycle events fall
    back to node-level association for compatibility.
    """
    generation = next(
        item for item in rollout.annotations.generations if item.generation_id == generation_id
    )
    active_generation_by_invocation: dict[str, str] = {}
    generation_lifecycle_seen = False
    edge_ids: set[str] = set()
    for event in rollout.execution.events:
        event_type = str(event.get("event_type") or "")
        invocation_id = str(event.get("invocation_id") or "")
        if event_type in {"student_generation_started", "student_generation_completed"}:
            if invocation_id == generation.node_id:
                generation_lifecycle_seen = True
            active_generation_by_invocation[invocation_id] = str(event.get("generation_id") or "")
        elif (
            event_type == "helper_question_generated"
            and active_generation_by_invocation.get(invocation_id) == generation_id
        ):
            edge_ids.add(str(event["subcall_id"]))
    if generation_lifecycle_seen:
        return frozenset(edge_ids)
    return frozenset(
        edge.edge_id for edge in rollout.execution.edges if edge.parent_id == generation.node_id
    )


def render_rubric_conditioning(
    rollout: AnnotatedRollout, generation_id: str, assessments: tuple[ScopedAssessment, ...]
) -> str:
    """Render only the rubric feedback for helper questions from one generation."""
    edge_ids = generation_edge_ids(rollout, generation_id)
    lines: list[str] = []
    for assessment in assessments:
        if not (set(assessment.focal_edge_ids) & edge_ids):
            continue
        rubric = (assessment.content or {}).get("rubric") or {}
        guidance = str(rubric.get("improved_question_guidance") or "").strip()
        missing = str(rubric.get("what_was_missing") or "").strip()
        if guidance or missing:
            lines.append(f"- ask instead: {guidance} (was missing: {missing})")
    if not lines:
        return ""
    return "Feedback on your previous helper questions — revise them:\n" + "\n".join(lines)


class TransformersPolicyScoreProvider:
    """Recompute differentiable student logits for exactly the selected sampled tokens.

    For each rollout it reconstructs the sampled generation, re-runs the policy with gradients to
    obtain logits over the continuation, and keeps only the rows at the selected token positions.
    The result feeds the SDPO loss, which expects ``[selected_tokens, vocabulary]`` logits keyed by
    rollout id.
    """

    def __init__(self, policy: TrainablePolicy) -> None:
        self.policy = policy

    def score(
        self,
        objective: str,
        capabilities: ObjectiveCapabilities,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, TokenSelectionResult],
    ) -> PolicyScoreBatch:
        """Return per-rollout student logits at the selected positions, with gradients enabled.

        Args:
            objective: Name of the requesting objective (unused; kept for the protocol).
            capabilities: Declared objective capabilities (unused for SDPO scoring).
            rollouts: Rollouts to score.
            selections: Per-rollout token selection identifying which positions to score.

        Returns:
            A ``PolicyScoreBatch`` mapping each rollout id to its ``[selected_tokens, vocabulary]``
            logits tensor.

        Raises:
            ValueError: If the policy does not return logits.
        """
        scores: dict[str, object] = {}
        for rollout in rollouts:
            selection = selections[rollout.rollout_id].durable
            generation_logits = []
            for generation_id, positions in selected_generation_positions(selection):
                sampled = reconstruct_generation(rollout, generation_id, self.policy)
                policy_score = self.policy.score_sampled_ids(
                    sampled,
                    require_grad=True,
                    return_logits=True,
                    return_logprobs=False,
                    positions=positions,
                )
                if policy_score.logits is None:
                    raise ValueError("policy must return logits for SDPO scoring")
                generation_logits.append(policy_score.logits)
            torch = __import__("torch")
            scores[rollout.rollout_id] = torch.cat(generation_logits, dim=0)
        return PolicyScoreBatch(policy_scores=scores)


class SelfDistillationTeacherTargetProvider:
    """Build top-k+tail teacher targets from the current policy held under no-grad."""

    def __init__(self, policy: TrainablePolicy, *, top_k: int, teacher_version: int = 0) -> None:
        self.policy = policy
        self.top_k = top_k
        self.teacher_version = teacher_version

    def build(
        self,
        objective: str,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, TokenSelectionResult],
        feedback: FeedbackBundle,
    ) -> dict[str, TeacherTarget]:
        projections = feedback.projections
        configuration_fingerprint = hashlib.sha256(
            json.dumps(
                {"top_k": self.top_k, "teacher_version": self.teacher_version}, sort_keys=True
            ).encode()
        ).hexdigest()
        targets: dict[str, TeacherTarget] = {}
        for rollout in rollouts:
            selection = selections[rollout.rollout_id].durable
            groups = selected_generation_positions(selection)
            selected_ids: list[int] = []
            selected_positions: list[int] = []
            selected_generation_ids: list[str] = []
            topk_token_ids: list[tuple[int, ...]] = []
            topk_logprobs: list[tuple[float, ...]] = []
            tail_logprobs: list[float] = []
            for generation_id, positions in groups:
                sampled = reconstruct_generation(rollout, generation_id, self.policy)
                # Condition the teacher only on feedback for helpers emitted by this generation.
                conditioning = render_rubric_conditioning(
                    rollout, generation_id, feedback.local_assessments
                )
                if conditioning:
                    prefix = tuple(self.policy.tokenize(conditioning + "\n"))
                    if prefix:
                        sampled = replace(
                            sampled, prompt_token_ids=prefix + sampled.prompt_token_ids
                        )
                policy_score = self.policy.score_sampled_ids(
                    sampled,
                    require_grad=False,
                    return_logits=True,
                    return_logprobs=False,
                    positions=positions,
                )
                if policy_score.logits is None:
                    raise ValueError("policy must return logits for SDPO teacher targets")
                topk = extract_topk_teacher_target(
                    policy_score.logits,
                    top_k=self.top_k,
                    teacher_version=self.teacher_version,
                    tokenizer_fingerprint=self.policy.tokenizer_identity.resolved_fingerprint,
                )
                selected_ids.extend(sampled.token_ids[position] for position in positions)
                selected_positions.extend(positions)
                selected_generation_ids.extend(generation_id for _ in positions)
                topk_token_ids.extend(topk.token_ids)
                topk_logprobs.extend(topk.logprobs)
                tail_logprobs.extend(topk.tail_logprobs)
            target_identity = {
                "rollout": rollout.rollout_id,
                "generation_positions": groups,
                "teacher": self.policy.identity.resolved_fingerprint,
                "configuration": configuration_fingerprint,
            }
            target_id = hashlib.sha256(
                json.dumps(target_identity, sort_keys=True).encode()
            ).hexdigest()
            targets[rollout.rollout_id] = TeacherTarget(
                target_id=target_id,
                rollout_id=rollout.rollout_id,
                generation_id=groups[0][0],
                selected_generation_ids=tuple(selected_generation_ids),
                selected_token_ids=tuple(selected_ids),
                selected_positions=tuple(selected_positions),
                topk_token_ids=tuple(topk_token_ids),
                topk_logprobs=tuple(topk_logprobs),
                tail_logprob_mass=tuple(tail_logprobs),
                teacher_fingerprint=self.policy.identity.resolved_fingerprint,
                tokenizer_fingerprint=self.policy.tokenizer_identity.resolved_fingerprint,
                feedback_projection_ids=tuple(item.projection_id for item in projections),
                judge_view_fingerprints=tuple(
                    fingerprint for item in projections for fingerprint in item.view_fingerprints
                ),
                feedback_visibility=tuple(item.visibility.value for item in projections),
                configuration_fingerprint=configuration_fingerprint,
            )
        return targets


class JudgeFeedbackProvider:
    """Turn traced helper-question edges into scoped judge assessments.

    For every rollout, every requested assessment scope, and every traced edge, it builds an
    ID-addressed ``JudgeView`` and asks the judge to score it, collecting the results into a single
    ``FeedbackBundle`` consumed downstream by teacher-target construction and artifact writing.
    """

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def assess(
        self,
        record: object,
        rollouts: tuple[AnnotatedRollout, ...],
        scopes: frozenset[AssessmentScope],
    ) -> FeedbackBundle:
        """Assess every traced edge of each rollout under each requested scope.

        Args:
            record: Dataset record for the rollouts (unused; kept for the protocol).
            rollouts: Rollouts whose edges are assessed.
            scopes: Assessment scopes to evaluate, processed in a deterministic order.

        Returns:
            A ``FeedbackBundle`` holding one local assessment per (rollout, scope, edge).
        """
        assessments = []
        for rollout in rollouts:
            for scope in sorted(scopes, key=lambda item: item.value):
                for edge in rollout.execution.edges:
                    view = build_judge_view(rollout, scope=scope, focal_edge_ids=(edge.edge_id,))
                    assessments.append(self.judge.assess(view))
        return FeedbackBundle(local_assessments=tuple(assessments))


def build_policy_score_provider(policy: TrainablePolicy) -> TransformersPolicyScoreProvider:
    """Wrap the shared policy in the provider that recomputes differentiable student scores."""
    return TransformersPolicyScoreProvider(policy)


def build_feedback_provider(judge: Judge) -> JudgeFeedbackProvider:
    """Wrap the judge in the provider that assesses traced edges into a FeedbackBundle."""
    return JudgeFeedbackProvider(judge)


__all__ = [
    "JudgeFeedbackProvider",
    "SelfDistillationTeacherTargetProvider",
    "TransformersPolicyScoreProvider",
    "build_feedback_provider",
    "build_policy_score_provider",
    "generation_edge_ids",
    "reconstruct_generation",
    "render_rubric_conditioning",
    "selected_generation_positions",
]
