"""Whole-recursive-policy evaluation configuration."""

from __future__ import annotations

from pydantic import Field, model_validator

from rlm_train.spec.models import ImmutableSpec


class EvaluationSpec(ImmutableSpec):
    recursive_policy: bool = True
    benchmarks: tuple[str, ...] = ()
    samples_per_problem: int = Field(default=1, gt=0)
    checkpoint_steps: tuple[int, ...] = ()
    base_seed: int = Field(default=0, ge=0)
    observer_judge: bool = False

    @model_validator(mode="after")
    def validate_recursive_policy(self) -> EvaluationSpec:
        if not self.recursive_policy:
            raise ValueError("evaluation must execute the full recursive policy")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint_steps must be unique and increasing")
        return self


__all__ = ["EvaluationSpec"]
