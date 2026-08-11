"""Gram-matrix representation alignment."""

from rlm_train.gram.calculate_loss import calculate_loss, gram_matrix_loss
from rlm_train.gram.choose_hidden_states import (
    choose_hidden_state_layers,
    choose_hidden_state_tokens,
)
from rlm_train.gram.settings import GramSettings

__all__ = [
    "GramSettings",
    "calculate_loss",
    "choose_hidden_state_layers",
    "choose_hidden_state_tokens",
    "gram_matrix_loss",
]
