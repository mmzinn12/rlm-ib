"""Trainable-policy generation and exact sampled-ID scoring contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from rlm_train.models.identity import PolicyIdentity, TokenizerIdentity


@dataclass(frozen=True)
class SampledGeneration:
    text: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    policy: PolicyIdentity
    tokenizer: TokenizerIdentity
    behavior_logprobs: Any | None = None
    hidden_states: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids or not self.token_ids:
            raise ValueError("sampled generations require prompt and continuation token IDs")
        if len(self.token_ids) != len(self.token_offsets):
            raise ValueError("sampled token IDs and character offsets must align")


@dataclass(frozen=True)
class PolicyScore:
    token_ids: tuple[int, ...]
    logprobs: Any | None = None
    logits: Any | None = None
    hidden_states: Any | None = None


class TrainablePolicy(Protocol):
    @property
    def identity(self) -> PolicyIdentity: ...

    @property
    def tokenizer_identity(self) -> TokenizerIdentity: ...

    def score_sampled_ids(
        self,
        generation: SampledGeneration,
        *,
        require_grad: bool,
        return_logits: bool = False,
        return_logprobs: bool = True,
        positions: tuple[int, ...] | None = None,
        capture_hidden_states: bool = False,
    ) -> PolicyScore: ...

    def tokenize(self, text: str) -> tuple[int, ...]: ...

    def trainable_parameters(self) -> Iterable[Any]: ...


__all__ = [
    "PolicyScore",
    "SampledGeneration",
    "TrainablePolicy",
]
