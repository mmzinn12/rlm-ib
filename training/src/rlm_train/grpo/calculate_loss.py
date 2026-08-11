"""Calculate the clipped group-relative student loss."""

from rlm_train.objectives.grpo.objective import grpo_policy_loss

calculate_loss = grpo_policy_loss

__all__ = ["calculate_loss"]
