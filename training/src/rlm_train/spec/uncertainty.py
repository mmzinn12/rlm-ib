"""Declarative semantic uncertainty configuration."""

from __future__ import annotations

from pydantic import Field, model_validator

from rlm_train.spec.models import ImmutableSpec


class UncertaintySpec(ImmutableSpec):
    enabled: bool = False
    estimator: str = "semantic_entropy"
    estimator_version: str = "semantic-entropy-v1-natural-log"
    sample_count: int = Field(default=10, ge=2)
    temperature: float = Field(default=0.5, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=32, gt=0)
    prompt_version: str = "direct-answer-v1"
    equivalence_provider: str = "transformers_nli"
    equivalence_model: str = "microsoft/deberta-large-mnli"
    equivalence_model_revision: str | None = None
    max_edges_per_rollout: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_configuration(self) -> UncertaintySpec:
        if self.estimator != "semantic_entropy":
            raise ValueError("only probability-weighted semantic_entropy is implemented")
        if self.prompt_version != "direct-answer-v1":
            raise ValueError("only direct-answer-v1 uncertainty prompts are implemented")
        if self.equivalence_provider not in {"transformers_nli", "exact_match"}:
            raise ValueError("equivalence_provider must be transformers_nli or exact_match")
        if self.enabled and self.equivalence_provider == "transformers_nli":
            revision = self.equivalence_model_revision
            if revision is None or revision in {"main", "latest", "default"}:
                raise ValueError("enabled Transformers NLI uncertainty requires a pinned revision")
        return self


__all__ = ["UncertaintySpec"]
