"""Generation and prompt-formatting settings."""

from rlm_train.models.transformers_runtime import GenerationConfig


class GenerationSettings(GenerationConfig):
    """Exact prompt-formatting and sampling settings."""


__all__ = ["GenerationSettings"]
