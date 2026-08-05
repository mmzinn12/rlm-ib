"""Serializable generic trainer state."""

from pydantic import BaseModel, ConfigDict, Field


class TrainerState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer_step: int = Field(default=0, ge=0)
    examples_seen: int = Field(default=0, ge=0)
    run_spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


__all__ = ["TrainerState"]
