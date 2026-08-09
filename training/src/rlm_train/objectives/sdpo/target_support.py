"""Detached top-k-plus-tail teacher distributions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TopKTeacherTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_ids: tuple[tuple[int, ...], ...]
    logprobs: tuple[tuple[float, ...], ...]
    tail_logprobs: tuple[float, ...]
    teacher_version: int = Field(ge=0)
    tokenizer_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_alignment(self) -> TopKTeacherTarget:
        if not self.token_ids or len(self.token_ids) != len(self.logprobs):
            raise ValueError("teacher token IDs and logprobs must align")
        if len(self.tail_logprobs) != len(self.token_ids):
            raise ValueError("teacher tail probabilities must align with token positions")
        if any(
            not ids or len(ids) != len(values)
            for ids, values in zip(self.token_ids, self.logprobs, strict=True)
        ):
            raise ValueError("teacher token IDs and logprobs must align")
        return self


def extract_topk_teacher_target(
    teacher_logits: Any,
    *,
    top_k: int,
    teacher_version: int,
    tokenizer_fingerprint: str,
) -> TopKTeacherTarget:
    """Detach logits and retain normalized top-k values plus exact tail mass."""
    try:
        torch = __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for teacher target extraction") from exc
    if not isinstance(teacher_logits, torch.Tensor) or teacher_logits.ndim != 2:
        raise TypeError("teacher logits must be a [tokens, vocabulary] PyTorch tensor")
    token_count, vocabulary_size = teacher_logits.shape
    if token_count == 0 or top_k <= 0 or top_k >= vocabulary_size:
        raise ValueError("top_k must leave a non-empty teacher tail")
    if teacher_version < 0 or not tokenizer_fingerprint.strip():
        raise ValueError("teacher identity fields are invalid")
    if not torch.isfinite(teacher_logits).all().item():
        raise ValueError("teacher logits must be finite")
    with torch.no_grad():
        # float64 keeps the compact top-k + tail distribution summing to one within 1e-8.
        logprobs = torch.log_softmax(teacher_logits.detach().to(dtype=torch.float64), dim=-1)
        topk_logprobs, token_ids = torch.topk(logprobs, k=top_k, dim=-1)
        tail_values = logprobs.clone()
        tail_values.scatter_(dim=-1, index=token_ids, value=-torch.inf)
        tail_logprobs = torch.logsumexp(tail_values, dim=-1)
        compact_logprobs = torch.cat((topk_logprobs, tail_logprobs.unsqueeze(-1)), dim=-1)
        compact_normalizer = torch.logsumexp(compact_logprobs, dim=-1, keepdim=True)
        topk_logprobs = topk_logprobs - compact_normalizer
        tail_logprobs = tail_logprobs - compact_normalizer.squeeze(-1)
    return TopKTeacherTarget(
        token_ids=tuple(tuple(int(value) for value in row) for row in token_ids.cpu().tolist()),
        logprobs=tuple(
            tuple(float(value) for value in row) for row in topk_logprobs.cpu().tolist()
        ),
        tail_logprobs=tuple(float(value) for value in tail_logprobs.cpu().tolist()),
        teacher_version=teacher_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


__all__ = ["TopKTeacherTarget", "extract_topk_teacher_target"]
