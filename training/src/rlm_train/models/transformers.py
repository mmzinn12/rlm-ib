"""Transformers implementation of the trainable policy protocol."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlm_train.models.identity import PolicyIdentity, TokenizerIdentity
from rlm_train.models.protocol import (
    GenerationRequest,
    PolicyScore,
    SampledGeneration,
)
from rlm_train.models.transformers_runtime import (
    GenerationConfig,
    PromptFormatter,
    TransformersCompletionAdapter,
    TransformersResponseGenerator,
    continuation_logprobs,
    decode_token_offsets,
    score_continuation_logits,
)


class TransformersPolicy(TransformersCompletionAdapter):
    """Use one local Transformers policy for root, plain, and recursive calls."""

    def __init__(
        self,
        generator: TransformersResponseGenerator,
        *,
        identity: PolicyIdentity,
        tokenizer_identity: TokenizerIdentity,
        base_seed: int,
    ) -> None:
        super().__init__(
            generator,
            model_name=identity.component_id,
            base_seed=base_seed,
        )
        self._identity = identity
        self._tokenizer_identity = tokenizer_identity

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def tokenizer_identity(self) -> TokenizerIdentity:
        return self._tokenizer_identity

    @property
    def policy_owner(self) -> str:
        return self.identity.policy_owner

    def generate(self, request: GenerationRequest) -> SampledGeneration:
        result = self.generator.generate_tokenized(request.prompt, seed=request.seed)
        return SampledGeneration(
            text=result.response,
            prompt_token_ids=result.prompt_token_ids,
            token_ids=result.continuation_token_ids,
            token_offsets=result.continuation_token_offsets,
            policy=self.identity,
            tokenizer=self.tokenizer_identity,
            metadata=result.sampling_metadata or {},
        )

    def score_sampled_ids(
        self,
        generation: SampledGeneration,
        *,
        require_grad: bool,
        return_logits: bool = False,
        capture_hidden_states: bool = False,
    ) -> PolicyScore:
        if capture_hidden_states:
            raise NotImplementedError("hidden-state capture is configured by the Gram adapter")
        logprobs = continuation_logprobs(
            self.generator.model,
            prompt_token_ids=generation.prompt_token_ids,
            continuation_token_ids=generation.token_ids,
            require_grad=require_grad,
        )
        logits = None
        if return_logits:
            logits = score_continuation_logits(
                self.generator.model,
                prompt_token_ids=generation.prompt_token_ids,
                continuation_token_ids=generation.token_ids,
                require_grad=require_grad,
            )
        return PolicyScore(token_ids=generation.token_ids, logprobs=logprobs, logits=logits)

    def trainable_parameters(self) -> Iterable[Any]:
        return (
            parameter for parameter in self.generator.model.parameters() if parameter.requires_grad
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        encoded = self.generator.tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(token_id) for token_id in encoded)


__all__ = [
    "PromptFormatter",
    "GenerationConfig",
    "TransformersPolicy",
    "TransformersResponseGenerator",
    "continuation_logprobs",
    "decode_token_offsets",
    "score_continuation_logits",
]
