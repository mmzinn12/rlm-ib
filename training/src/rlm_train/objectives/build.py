"""Build an ObjectiveComposer from the declarative objectives specification."""

from __future__ import annotations

from rlm_train.objectives.composer import ObjectiveComposer
from rlm_train.objectives.sdpo.loss import build_sdpo_objective
from rlm_train.spec.objectives import ObjectivesSpec


def build_objective_composer(spec: ObjectivesSpec) -> ObjectiveComposer:
    objectives: dict[str, tuple[float, object]] = {}
    if spec.sdpo.enabled:
        objectives["sdpo"] = (spec.sdpo.weight, build_sdpo_objective(spec.sdpo))
    if spec.grpo.enabled:
        raise NotImplementedError("GRPO objective assembly is not wired yet")
    if spec.gram.enabled:
        raise NotImplementedError("Gram objective assembly is not wired yet")
    if not objectives:
        raise ValueError("at least one objective must be enabled to build a composer")
    return ObjectiveComposer(objectives)


__all__ = ["build_objective_composer"]
