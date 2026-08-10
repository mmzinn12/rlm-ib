"""Rollout engine construction from a RunSpec and an injected policy.

The rollout engine wraps a trainable policy in the RLM execution loop that produces trajectories
for both training and evaluation. ``build_rollout_engine`` is the single entry point that reads
the rollout configuration off a ``RunSpec`` and binds it to the shared policy instance.
"""

from __future__ import annotations

from typing import Any

from rlm_train.rollouts.rlm_engine import RLMRolloutEngine
from rlm_train.spec import RunSpec


def build_rollout_engine(
    run: RunSpec,
    *,
    policy: Any,
    backend: str = "openai",
    environment_kwargs: dict[str, Any] | None = None,
) -> RLMRolloutEngine:
    """Bind the shared policy to the RLM rollout engine described by the RunSpec.

    Args:
        run: Run specification; ``rollout`` supplies the engine config and ``student`` the owner.
        policy: The shared trainable policy used to generate root and recursive completions.
        backend: RLM client backend identifier passed through to the engine.
        environment_kwargs: Optional overrides forwarded to the RLM execution environment.

    Returns:
        An ``RLMRolloutEngine`` ready to execute rollout requests.
    """
    return RLMRolloutEngine(
        policy=policy,
        policy_owner=run.student.resolved_policy_owner,
        spec=run.rollout,
        backend=backend,
        environment_kwargs=environment_kwargs,
    )


__all__ = ["build_rollout_engine"]
