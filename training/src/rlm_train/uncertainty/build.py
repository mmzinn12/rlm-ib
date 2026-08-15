"""Construct semantic uncertainty collaborators from RunSpec."""

from __future__ import annotations

from typing import Any

from rlm_train.engine.uncertainty_provider import SemanticEntropyUncertaintyProvider
from rlm_train.spec import RunSpec
from rlm_train.uncertainty.sampling import TransformersAnswerSampler
from rlm_train.uncertainty.semantic_entropy import ProbabilityWeightedSemanticEntropyEstimator
from rlm_train.uncertainty.semantic_equivalence import (
    ExactMatchEquivalenceClassifier,
    TransformersNLIEquivalenceClassifier,
)


def build_uncertainty_provider(run: RunSpec, *, policy: Any) -> Any | None:
    if not run.uncertainty.enabled:
        return None
    classifier = (
        ExactMatchEquivalenceClassifier()
        if run.uncertainty.equivalence_provider == "exact_match"
        else TransformersNLIEquivalenceClassifier(
            run.uncertainty.equivalence_model,
            run.uncertainty.equivalence_model_revision or "",
        )
    )
    return SemanticEntropyUncertaintyProvider(
        sampler=TransformersAnswerSampler(
            policy,
            max_prompt_tokens=run.student.generation.max_prompt_tokens,
            use_chat_template=run.student.generation.use_chat_template,
            allow_prompt_truncation=run.student.generation.allow_prompt_truncation,
        ),
        classifier=classifier,
        estimator=ProbabilityWeightedSemanticEntropyEstimator(
            estimator_version=run.uncertainty.estimator_version
        ),
        configuration=run.uncertainty,
        run_seed=run.runtime.seed,
    )


__all__ = ["build_uncertainty_provider"]
