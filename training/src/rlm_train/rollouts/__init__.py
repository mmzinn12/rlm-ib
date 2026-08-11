"""Canonical rollout execution, recording, semantics, and selection."""

from rlm_train.rollouts.protocol import RolloutEngine, RolloutRequest, RolloutResult
from rlm_train.rollouts.recorder import RolloutRecorder
from rlm_train.rollouts.rlm_engine import RLMRolloutEngine
from rlm_train.rollouts.selectors import TokenSelectionResult, select_tokens
from rlm_train.rollouts.token_alignment import contained_token_range_for_characters

__all__ = [
    "RLMRolloutEngine",
    "RolloutEngine",
    "RolloutRecorder",
    "RolloutRequest",
    "RolloutResult",
    "TokenSelectionResult",
    "contained_token_range_for_characters",
    "select_tokens",
]
