"""Declarative student, judge, and teacher model specifications."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class ImmutableSpec(BaseModel):
    """Frozen configuration boundary shared by every run specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationSpec(ImmutableSpec):
    """Prompt-length and sampling settings for the student's text generation.

    Attributes:
        max_prompt_tokens: Max tokens allowed in a formatted prompt before generation.
        max_new_tokens: Max tokens sampled per generation.
        temperature: Sampling temperature (used when ``do_sample``).
        top_p: Nucleus-sampling probability mass (used when ``do_sample``).
        do_sample: Sample when ``True``; greedy decode when ``False``.
        use_chat_template: Format prompts with the tokenizer's chat template.
        allow_prompt_truncation: Truncate over-long prompts instead of raising.
    """

    max_prompt_tokens: int = Field(default=512, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    do_sample: bool = True
    use_chat_template: bool = True
    allow_prompt_truncation: bool = False


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
    generation: GenerationSpec = Field(default_factory=GenerationSpec)

    @property
    def resolved_policy_owner(self) -> str:
        return self.policy_owner or f"student:{self.model_id}:{self.revision or 'default'}"


class JudgeMode(StrEnum):
    """Select the structured assessment contract used by an LLM judge."""

    CATEGORICAL = "categorical"
    FULL = "full"


class JudgeSpec(ImmutableSpec):
    provider: str = Field(default="fake", min_length=1)
    model: str = Field(default="deterministic-fake", min_length=1)
    model_revision: str = Field(default="v1", min_length=1)
    schema_name: str = Field(default="trajectory-v1", alias="schema", min_length=1)
    prompt_version: str = Field(default="v1", min_length=1)
    mode: JudgeMode = JudgeMode.CATEGORICAL
    api_key_environment: str = Field(default="OPENAI_API_KEY", min_length=1)
    base_url: AnyHttpUrl | None = None
    max_attempts: int = Field(default=3, ge=1, le=5)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def validate_provider(self) -> JudgeSpec:
        if self.provider == "openai" and self.model == "deterministic-fake":
            raise ValueError("openai judge provider requires a real model route")
        return self


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
    "GenerationSpec",
    "ImmutableSpec",
    "JudgeMode",
    "JudgeSpec",
    "StudentSpec",
    "TeacherSpec",
    "TeacherStrategy",
]
