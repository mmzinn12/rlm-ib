"""Contract for the model whose parameters are updated by training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from rlm_train.generation.generated_text import GeneratedText
from rlm_train.student.model_info import StudentModelInfo, TokenizerInfo
from rlm_train.student.score_tokens import TokenPredictions


class TrainableStudent(Protocol):
    @property
    def model_info(self) -> StudentModelInfo: ...

    @property
    def tokenizer_info(self) -> TokenizerInfo: ...

    def score_tokens(
        self,
        generated_text: GeneratedText,
        *,
        with_gradients: bool,
        return_logits: bool = False,
        return_logprobs: bool = True,
        positions: tuple[int, ...] | None = None,
        capture_hidden_states: bool = False,
    ) -> TokenPredictions: ...

    def format_prompt(self, messages: list[dict[str, str]]) -> tuple[int, ...]: ...

    def trainable_parameters(self) -> Iterable[Any]: ...


__all__ = ["TrainableStudent"]
