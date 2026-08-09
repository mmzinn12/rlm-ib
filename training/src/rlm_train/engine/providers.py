"""Concrete training providers: differentiable student scores, judge feedback, teacher targets."""

from __future__ import annotations

import hashlib
import json

from rlm_train.engine.trainer import PolicyScoreBatch
from rlm_train.feedback.schema import FeedbackBundle
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


class TransformersPolicyScoreProvider:
    """Recompute differentiable logits for exactly the selected sampled tokens."""

    def __init__(self, policy: TrainablePolicy) -> None:
        self.policy = policy

    def score(
        self,
        objective: str,
        capabilities: ObjectiveCapabilities,
        rollouts: tuple[AnnotatedRollout, ...],
        selections: dict[str, TokenSelectionResult],
    ) -> PolicyScoreBatch:
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
            sampled = reconstruct_generation(rollout, generation_id, self.policy)
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
    """Assess each traced helper-question edge under the requested scopes."""

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def assess(
        self,
        record: object,
        rollouts: tuple[AnnotatedRollout, ...],
        scopes: frozenset[AssessmentScope],
    ) -> FeedbackBundle:
        assessments = []
        for rollout in rollouts:
            for scope in sorted(scopes, key=lambda item: item.value):
                for edge in rollout.execution.edges:
                    view = build_judge_view(rollout, scope=scope, focal_edge_ids=(edge.edge_id,))
                    assessments.append(self.judge.assess(view))
        return FeedbackBundle(local_assessments=tuple(assessments))


__all__ = [
    "JudgeFeedbackProvider",
    "SelfDistillationTeacherTargetProvider",
    "TransformersPolicyScoreProvider",
    "reconstruct_generation",
    "selected_positions",
]
