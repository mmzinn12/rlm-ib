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


def selected_positions(selection: ObjectiveSelection) -> tuple[int, ...]:
    generation_id = selection.ranges[0].generation_id
    positions = tuple(
        position
        for item in selection.ranges
        if item.generation_id == generation_id
        for position in range(item.token_start, item.token_end)
    )
    if not positions:
        raise ValueError("objective selection has no selected token positions")
    return positions


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


def render_rubric_conditioning(
    rollout: AnnotatedRollout, node_id: str, assessments: tuple[ScopedAssessment, ...]
) -> str:
    """Render the judge's rubric feedback for a node's helper questions into a revision hint."""
    edge_ids = {edge.edge_id for edge in rollout.execution.edges if edge.parent_id == node_id}
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
        torch = __import__("torch")
        scores: dict[str, object] = {}
        for rollout in rollouts:
            selection = selections[rollout.rollout_id].durable
            generation_id = selection.ranges[0].generation_id
            positions = selected_positions(selection)
            sampled = reconstruct_generation(rollout, generation_id, self.policy)
            policy_score = self.policy.score_sampled_ids(
                sampled, require_grad=True, return_logits=True
            )
            if policy_score.logits is None:
                raise ValueError("policy must return logits for SDPO scoring")
            index = torch.tensor(positions, dtype=torch.long, device=policy_score.logits.device)
            scores[rollout.rollout_id] = policy_score.logits.index_select(0, index)
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
        torch = __import__("torch")
        projections = feedback.projections
        configuration_fingerprint = hashlib.sha256(
            json.dumps(
                {"top_k": self.top_k, "teacher_version": self.teacher_version}, sort_keys=True
            ).encode()
        ).hexdigest()
        targets: dict[str, TeacherTarget] = {}
        for rollout in rollouts:
            selection = selections[rollout.rollout_id].durable
            generation_id = selection.ranges[0].generation_id
            positions = selected_positions(selection)
            generation_record = next(
                item
                for item in rollout.annotations.generations
                if item.generation_id == generation_id
            )
            sampled = reconstruct_generation(rollout, generation_id, self.policy)
            # Condition the teacher on the rubric feedback so it differs from the student.
            conditioning = render_rubric_conditioning(
                rollout, generation_record.node_id, feedback.local_assessments
            )
            if conditioning:
                prefix = tuple(self.policy.tokenize(conditioning + "\n"))
                if prefix:
                    sampled = replace(
                        sampled, prompt_token_ids=prefix + sampled.prompt_token_ids
                    )
            policy_score = self.policy.score_sampled_ids(
                sampled, require_grad=False, return_logits=True
            )
            index = torch.tensor(positions, dtype=torch.long, device=policy_score.logits.device)
            teacher_logits = policy_score.logits.index_select(0, index)
            topk = extract_topk_teacher_target(
                teacher_logits,
                top_k=self.top_k,
                teacher_version=self.teacher_version,
                tokenizer_fingerprint=self.policy.tokenizer_identity.resolved_fingerprint,
            )
            selected_ids = tuple(sampled.token_ids[position] for position in positions)
            target_identity = {
                "rollout": rollout.rollout_id,
                "generation": generation_id,
                "positions": positions,
                "teacher": self.policy.identity.resolved_fingerprint,
                "configuration": configuration_fingerprint,
            }
            target_id = hashlib.sha256(
                json.dumps(target_identity, sort_keys=True).encode()
            ).hexdigest()
            targets[rollout.rollout_id] = TeacherTarget(
                target_id=target_id,
                rollout_id=rollout.rollout_id,
                generation_id=generation_id,
                selected_token_ids=selected_ids,
                selected_positions=positions,
                topk_token_ids=topk.token_ids,
                topk_logprobs=topk.logprobs,
                tail_logprob_mass=topk.tail_logprobs,
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
    "reconstruct_generation",
    "render_rubric_conditioning",
    "selected_positions",
]
