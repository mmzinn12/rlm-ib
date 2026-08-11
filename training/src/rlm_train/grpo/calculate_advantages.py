"""Calculate normalized group-relative advantages."""

from rlm_train.objectives.grpo.advantages import group_relative_advantages

calculate_advantages = group_relative_advantages

__all__ = ["calculate_advantages"]
