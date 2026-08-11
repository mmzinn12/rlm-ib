"""Choose model layers and token positions for Gram alignment."""

from rlm_train.objectives.gram.sampling import sample_token_positions
from rlm_train.objectives.gram.selection import resolve_layer_selection

choose_hidden_state_layers = resolve_layer_selection
choose_hidden_state_tokens = sample_token_positions

__all__ = ["choose_hidden_state_layers", "choose_hidden_state_tokens"]
