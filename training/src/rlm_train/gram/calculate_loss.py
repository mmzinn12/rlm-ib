"""Calculate single- or multi-layer Gram alignment losses."""

from rlm_train.objectives.gram.math import gram_matrix_loss, multi_layer_gram_loss

calculate_loss = multi_layer_gram_loss

__all__ = ["calculate_loss", "gram_matrix_loss"]
