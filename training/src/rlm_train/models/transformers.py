"""Transformers implementation of the trainable policy protocol."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rlm_train.models.identity import PolicyIdentity, TokenizerIdentity
from rlm_train.models.protocol import (
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

    def score_sampled_ids(
        self,
        generation: SampledGeneration,
        *,
        require_grad: bool,
        return_logits: bool = False,
        return_logprobs: bool = True,
        positions: tuple[int, ...] | None = None,
        capture_hidden_states: bool = False,
    ) -> PolicyScore:
        if capture_hidden_states:
            raise NotImplementedError("hidden-state capture is configured by the Gram adapter")
        if not return_logits and not return_logprobs:
            raise ValueError("policy scoring must request logits, logprobs, or both")
        logprobs = None
        logits = None
        if return_logits:
            logits = score_continuation_logits(
                self.generator.model,
                prompt_token_ids=generation.prompt_token_ids,
                continuation_token_ids=generation.token_ids,
                require_grad=require_grad,
                positions=positions,
            )
            if return_logprobs:
                logprobs = _gather_logprobs(logits, generation.token_ids, positions)
        elif return_logprobs:
            logprobs = continuation_logprobs(
                self.generator.model,
                prompt_token_ids=generation.prompt_token_ids,
                continuation_token_ids=generation.token_ids,
                require_grad=require_grad,
                positions=positions,
            )
        return PolicyScore(token_ids=generation.token_ids, logprobs=logprobs, logits=logits)

    def trainable_parameters(self) -> Iterable[Any]:
        return (
            parameter for parameter in self.generator.model.parameters() if parameter.requires_grad
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        encoded = self.generator.tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(token_id) for token_id in encoded)

    def save_pretrained(self, destination: str | Path) -> None:
        """Save model and tokenizer files loadable by the Transformers builders."""
        self.generator.model.save_pretrained(destination)
        self.generator.tokenizer.save_pretrained(destination)


def _gather_logprobs(
    logits: Any,
    continuation_token_ids: tuple[int, ...],
    positions: tuple[int, ...] | None,
) -> Any:
    torch = __import__("torch")
    selected_positions = positions or tuple(range(len(continuation_token_ids)))
    targets = torch.tensor(
        [continuation_token_ids[position] for position in selected_positions],
        dtype=torch.long,
        device=logits.device,
    )
    return (
        torch.log_softmax(logits.float(), dim=-1).gather(dim=-1, index=targets[:, None]).squeeze(-1)
    )


__all__ = [
    "PromptFormatter",
    "GenerationConfig",
    "TransformersPolicy",
    "TransformersResponseGenerator",
    "continuation_logprobs",
    "decode_token_offsets",
    "score_continuation_logits",
]
