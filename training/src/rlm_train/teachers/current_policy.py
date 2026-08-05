"""Feedback-conditioned current-policy self-teacher."""

from __future__ import annotations

import hashlib
import json

from rlm_train.feedback.schema import FeedbackProjection
from rlm_train.models.protocol import SampledGeneration, TrainablePolicy
from rlm_train.teachers.targets import TeacherTarget, tensor_values
from rlm_train.trajectory.schema import ObjectiveSelection


class CurrentPolicyTeacher:
    def __init__(self, policy: TrainablePolicy, *, configuration: dict[str, object] | None = None):
        self.policy = policy
        self.configuration = dict(configuration or {})

    def build_target(
        self,
        *,
        rollout_id: str,
        generation: SampledGeneration,
        selection: ObjectiveSelection,
        feedback: tuple[FeedbackProjection, ...],
    ) -> TeacherTarget:
        positions = tuple(
            position
            for item in selection.ranges
            if item.generation_id == selection.ranges[0].generation_id
            for position in range(item.token_start, item.token_end)
        )
        if not positions:
            raise ValueError("teacher target requires a non-empty objective selection")
        score = self.policy.score_sampled_ids(generation, require_grad=False)
        selected_ids = tuple(generation.token_ids[position] for position in positions)
        selected_logprobs = score.logprobs[list(positions)]
        config_json = json.dumps(self.configuration, sort_keys=True, separators=(",", ":"))
        config_fingerprint = hashlib.sha256(config_json.encode()).hexdigest()
        target_identity = {
            "rollout": rollout_id,
            "generation": selection.ranges[0].generation_id,
            "positions": positions,
            "teacher": self.policy.identity.resolved_fingerprint,
            "feedback": [item.projection_id for item in feedback],
            "configuration": config_fingerprint,
        }
        target_id = hashlib.sha256(
            json.dumps(target_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return TeacherTarget(
            target_id=target_id,
            rollout_id=rollout_id,
            generation_id=selection.ranges[0].generation_id,
            selected_token_ids=selected_ids,
            selected_positions=positions,
            teacher_logprobs=tensor_values(selected_logprobs),
            teacher_fingerprint=self.policy.identity.resolved_fingerprint,
            tokenizer_fingerprint=self.policy.tokenizer_identity.resolved_fingerprint,
            feedback_projection_ids=tuple(item.projection_id for item in feedback),
            judge_view_fingerprints=tuple(
                fingerprint for item in feedback for fingerprint in item.view_fingerprints
            ),
            feedback_visibility=tuple(item.visibility.value for item in feedback),
            configuration_fingerprint=config_fingerprint,
        )

    def after_optimizer_step(self) -> None:
        return None


__all__ = ["CurrentPolicyTeacher"]
