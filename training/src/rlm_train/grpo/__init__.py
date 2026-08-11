"""Group-relative training method."""

from rlm_train.grpo.calculate_advantages import calculate_advantages
from rlm_train.grpo.calculate_loss import calculate_loss
from rlm_train.grpo.settings import GRPOSettings

__all__ = ["GRPOSettings", "calculate_advantages", "calculate_loss"]
