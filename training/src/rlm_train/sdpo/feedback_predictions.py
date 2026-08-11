"""Detached feedback-conditioned predictions aligned to selected student tokens."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedbackPredictions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    prediction_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    selected_generation_ids: tuple[str, ...] = ()
    selected_token_ids: tuple[int, ...]
    selected_positions: tuple[int, ...]
    logprobs: tuple[float, ...] = ()
    topk_token_ids: tuple[tuple[int, ...], ...] = ()
    topk_logprobs: tuple[tuple[float, ...], ...] = ()
    tail_logprob_mass: tuple[float, ...] = ()
    student_fingerprint: str = Field(min_length=1)
    tokenizer_fingerprint: str = Field(min_length=1)
    feedback_assessment_ids: tuple[str, ...] = ()
    feedback_projection_ids: tuple[str, ...] = ()
    judge_view_fingerprints: tuple[str, ...] = ()
    feedback_visibility: tuple[str, ...] = ()
    settings_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_alignment(self) -> FeedbackPredictions:
        count = len(self.selected_token_ids)
        if count == 0 or len(self.selected_positions) != count:
            raise ValueError("feedback predictions require aligned selected IDs and positions")
        if self.selected_generation_ids and len(self.selected_generation_ids) != count:
            raise ValueError("selected generation IDs must align with selected tokens")
        if self.logprobs and len(self.logprobs) != count:
            raise ValueError("feedback log probabilities must align with selected tokens")
        if self.topk_token_ids:
            if len(self.topk_token_ids) != count or len(self.topk_logprobs) != count:
                raise ValueError("feedback top-k distributions must align with selected tokens")
            if len(self.tail_logprob_mass) != count:
                raise ValueError("feedback tail masses must align with selected tokens")
        return self


__all__ = ["FeedbackPredictions"]
