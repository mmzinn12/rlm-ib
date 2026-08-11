"""Exact-token Transformers generation."""

from typing import Any

from rlm_train.generation.format_prompt import PromptFormatter
from rlm_train.generation.settings import GenerationSettings
from rlm_train.models.transformers_runtime import (
    TokenGenerationResult,
    decode_token_offsets,
    derive_group_seed,
)
from rlm_train.models.transformers_runtime import (
    TransformersResponseGenerator as BaseTransformersGenerator,
)


class TransformersGenerator(BaseTransformersGenerator):
    """Generate text while retaining the model-returned continuation token IDs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        configuration: GenerationSettings,
        *,
        model_context_length: int,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            configuration,
            model_context_length=model_context_length,
        )
        self.formatter = PromptFormatter(tokenizer, configuration)


__all__ = [
    "TokenGenerationResult",
    "TransformersGenerator",
    "decode_token_offsets",
    "derive_group_seed",
]
