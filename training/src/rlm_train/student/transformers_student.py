"""Transformers implementation of the trainable-student contract."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rlm_train.generation.generate import TransformersGenerator
from rlm_train.generation.generated_text import GeneratedText
from rlm_train.generation.rlm_client import StudentRLMClient
from rlm_train.student.model_info import StudentModelInfo, TokenizerInfo
from rlm_train.student.score_tokens import (
    TokenPredictions,
    continuation_logprobs,
    score_continuation_logits,
)


class TransformersStudent(StudentRLMClient):
    """Share one model between RLM generation and differentiable token scoring."""

    def __init__(
        self,
        generator: TransformersGenerator,
        *,
        model_info: StudentModelInfo,
        tokenizer_info: TokenizerInfo,
        base_seed: int,
    ) -> None:
        super().__init__(generator, model_name=model_info.component_id, base_seed=base_seed)
        self._model_info = model_info
        self._tokenizer_info = tokenizer_info

    @property
    def model_info(self) -> StudentModelInfo:
        return self._model_info

    @property
    def tokenizer_info(self) -> TokenizerInfo:
        return self._tokenizer_info

    @property
    def policy_owner(self) -> str:
        return self.model_info.student_id

    def score_tokens(
        self,
        generated_text: GeneratedText,
        *,
        with_gradients: bool,
        return_logits: bool = False,
        return_logprobs: bool = True,
        positions: tuple[int, ...] | None = None,
        capture_hidden_states: bool = False,
    ) -> TokenPredictions:
        if capture_hidden_states:
            raise NotImplementedError("hidden-state capture is configured by the Gram method")
        if not return_logits and not return_logprobs:
            raise ValueError("token scoring must request logits, logprobs, or both")
        logits = None
        logprobs = None
        if return_logits:
            logits = score_continuation_logits(
                self.generator.model,
                prompt_token_ids=generated_text.prompt_token_ids,
                continuation_token_ids=generated_text.token_ids,
                with_gradients=with_gradients,
                positions=positions,
            )
            if return_logprobs:
                logprobs = gather_logprobs(logits, generated_text.token_ids, positions)
        elif return_logprobs:
            logprobs = continuation_logprobs(
                self.generator.model,
                prompt_token_ids=generated_text.prompt_token_ids,
                continuation_token_ids=generated_text.token_ids,
                with_gradients=with_gradients,
                positions=positions,
            )
        return TokenPredictions(
            token_ids=generated_text.token_ids,
            logits=logits,
            logprobs=logprobs,
        )

    def format_prompt(self, messages: list[dict[str, str]]) -> tuple[int, ...]:
        return self.generator.formatter.encode_prompt(messages)

    def trainable_parameters(self) -> Iterable[Any]:
        return (
            parameter for parameter in self.generator.model.parameters() if parameter.requires_grad
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        values = self.generator.tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(token_id) for token_id in values)

    def save_pretrained(self, destination: str | Path) -> None:
        self.generator.model.save_pretrained(destination)
        self.generator.tokenizer.save_pretrained(destination)


def gather_logprobs(
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


__all__ = ["TransformersStudent"]
