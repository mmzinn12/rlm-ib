"""Immutable teacher targets aligned to exact selected student token IDs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TeacherTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    selected_generation_ids: tuple[str, ...] = ()
    selected_token_ids: tuple[int, ...]
    selected_positions: tuple[int, ...]
    teacher_logprobs: tuple[float, ...] = ()
    topk_token_ids: tuple[tuple[int, ...], ...] = ()
    topk_logprobs: tuple[tuple[float, ...], ...] = ()
    tail_logprob_mass: tuple[float, ...] = ()
    teacher_fingerprint: str = Field(min_length=1)
    tokenizer_fingerprint: str = Field(min_length=1)
    feedback_projection_ids: tuple[str, ...] = ()
    judge_view_fingerprints: tuple[str, ...] = ()
    feedback_visibility: tuple[str, ...] = ()
    configuration_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_alignment(self) -> TeacherTarget:
        count = len(self.selected_token_ids)
        if count == 0 or len(self.selected_positions) != count:
            raise ValueError("teacher target requires aligned selected IDs and positions")
        if self.selected_generation_ids:
            if len(self.selected_generation_ids) != count:
                raise ValueError("selected generation IDs must align with selected tokens")
            if self.selected_generation_ids[0] != self.generation_id:
                raise ValueError("generation_id must identify the first selected generation")
        if self.teacher_logprobs and len(self.teacher_logprobs) != count:
            raise ValueError("teacher log probabilities must align with selected tokens")
        if self.topk_token_ids:
            if len(self.topk_token_ids) != count or len(self.topk_logprobs) != count:
                raise ValueError("teacher top-k distributions must align with selected tokens")
            if any(
                len(ids) != len(values)
                for ids, values in zip(self.topk_token_ids, self.topk_logprobs, strict=True)
            ):
                raise ValueError("teacher top-k IDs and log probabilities must align")
            if len(self.tail_logprob_mass) != count:
                raise ValueError("teacher tail masses must align with selected tokens")
        return self


def tensor_values(value: Any) -> tuple[float, ...]:
    detached = value.detach().float().cpu().reshape(-1)
    return tuple(float(item) for item in detached.tolist())


__all__ = ["TeacherTarget", "tensor_values"]
