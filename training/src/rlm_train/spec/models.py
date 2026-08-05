"""Declarative student, judge, and teacher model specifications."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableSpec(BaseModel):
    """Frozen configuration boundary shared by every run specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StudentSpec(ImmutableSpec):
    adapter: str = Field(default="transformers", min_length=1)
    model_id: str = Field(min_length=1)
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    trainable: bool = True
    policy_owner: str | None = None
    adapter_id: str | None = None
    trust_remote_code: bool = False

    @property
    def resolved_policy_owner(self) -> str:
        return self.policy_owner or f"student:{self.model_id}:{self.revision or 'default'}"


class JudgeSpec(ImmutableSpec):
    provider: str = Field(default="fake", min_length=1)
    model: str = Field(default="deterministic-fake", min_length=1)
    model_revision: str = Field(default="v1", min_length=1)
    schema_name: str = Field(default="trajectory-v1", alias="schema", min_length=1)
    prompt_version: str = Field(default="v1", min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TeacherStrategy(StrEnum):
    CURRENT_POLICY = "current_policy"
    FIXED = "fixed"
    EMA = "ema"
    EXTERNAL = "external"


class TeacherSpec(ImmutableSpec):
    strategy: TeacherStrategy = TeacherStrategy.CURRENT_POLICY
    model_id: str | None = None
    revision: str | None = None
    feedback_conditioning: bool = True
    ema_decay: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_strategy(self) -> TeacherSpec:
        if self.strategy is TeacherStrategy.FIXED and not self.model_id:
            raise ValueError("fixed teacher requires model_id")
        if self.strategy is TeacherStrategy.EMA and self.ema_decay is None:
            raise ValueError("EMA teacher requires ema_decay")
        if self.strategy is not TeacherStrategy.EMA and self.ema_decay is not None:
            raise ValueError("ema_decay is only valid for an EMA teacher")
        if self.strategy is TeacherStrategy.EXTERNAL and not self.model_id:
            raise ValueError("external teacher requires model_id")
        return self


__all__ = [
    "ImmutableSpec",
    "JudgeSpec",
    "StudentSpec",
    "TeacherSpec",
    "TeacherStrategy",
]
