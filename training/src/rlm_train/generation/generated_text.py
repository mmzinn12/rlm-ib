"""Exact model-returned text and token metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlm_train.student.model_info import StudentModelInfo, TokenizerInfo


@dataclass(frozen=True)
class GeneratedText:
    text: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]
    student: StudentModelInfo
    tokenizer: TokenizerInfo
    prompt_messages: tuple[dict[str, str], ...] = ()
    behavior_logprobs: Any | None = None
    hidden_states: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids or not self.token_ids:
            raise ValueError("generated text requires prompt and continuation token IDs")
        if len(self.token_ids) != len(self.token_offsets):
            raise ValueError("generated token IDs and character offsets must align")


__all__ = ["GeneratedText"]
