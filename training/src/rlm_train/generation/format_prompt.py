"""Chat-template prompt formatting."""

from rlm_train.models.transformers_runtime import PromptFormatter as BasePromptFormatter


class PromptFormatter(BasePromptFormatter):
    """Format ordinary and feedback-conditioned messages through one template."""


__all__ = ["PromptFormatter"]
