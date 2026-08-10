"""Build an ObjectiveComposer from the declarative objectives specification.

The objectives spec declares which training objectives (SDPO, GRPO, Gram) are enabled and their
weights. ``build_objective_composer`` is the single entry point that turns that declaration into
an ``ObjectiveComposer`` the trainer evaluates each step. Only SDPO is wired today; enabling an
unwired objective raises rather than silently doing nothing.
"""

from __future__ import annotations

from rlm_train.objectives.composer import ObjectiveComposer
from rlm_train.objectives.sdpo.loss import build_sdpo_objective
from rlm_train.spec.objectives import ObjectivesSpec


def build_objective_composer(spec: ObjectivesSpec) -> ObjectiveComposer:
    """Assemble the enabled objectives and their weights into a composer.

    Args:
        spec: Objectives specification listing which objectives are enabled and their weights.

    Returns:
        An ``ObjectiveComposer`` mapping each enabled objective name to its ``(weight, objective)``.

    Raises:
        ValueError: If no objective is enabled.
        NotImplementedError: If an enabled objective (GRPO or Gram) has no wired builder.
    """
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
