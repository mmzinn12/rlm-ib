"""GRPO numerical objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rlm_train.objectives.protocol import (
    ObjectiveBatch,
    ObjectiveCapabilities,
    ObjectiveResult,
)
from rlm_train.spec.objectives import GRPOSpec


@dataclass(frozen=True)
class PolicyLossResult:
    loss: Any
    approximate_kl: Any
    active_token_count: int
    ratio_mean: Any


def grpo_policy_loss(
    *,
    current_logprobs: Sequence[Any],
    behavior_logprobs: Sequence[Any],
    advantages: Sequence[float],
    masks: Sequence[Any],
    clip_epsilon: float,
    kl_coefficient: float = 0.0,
) -> PolicyLossResult:
    """Compute clipped grouped policy loss over selected continuation tokens."""
    torch = _torch()
    counts = {len(current_logprobs), len(behavior_logprobs), len(advantages), len(masks)}
    if len(counts) != 1 or not current_logprobs:
        raise ValueError("policy objective inputs must contain the same non-zero sample count")
    if clip_epsilon <= 0.0 or kl_coefficient < 0.0:
        raise ValueError("clip epsilon must be positive and KL coefficient non-negative")
    sample_losses: list[Any] = []
    token_kls: list[Any] = []
    token_ratios: list[Any] = []
    active_token_count = 0
    for current, behavior, advantage, mask_value in zip(
        current_logprobs,
        behavior_logprobs,
        advantages,
        masks,
        strict=True,
    ):
        if current.shape != behavior.shape or current.ndim != 1:
            raise ValueError("current and behavior log-probabilities must align per sample")
        mask = torch.as_tensor(mask_value, dtype=torch.bool, device=current.device)
        if mask.shape != current.shape:
            raise ValueError("policy mask must align with continuation log-probabilities")
        active = int(mask.sum().item())
        if active == 0:
            raise ValueError("each rollout requires at least one selected token")
        if not torch.isfinite(current).all().item() or not torch.isfinite(behavior).all().item():
            raise FloatingPointError("policy log-probabilities must be finite")
        log_ratio = current - behavior.detach().to(current.device)
        ratio = torch.exp(log_ratio)
        advantage_tensor = torch.as_tensor(float(advantage), device=current.device)
        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_tensor
        approximate_kl = torch.exp(log_ratio) - 1.0 - log_ratio
        token_loss = -torch.minimum(unclipped, clipped) + kl_coefficient * approximate_kl
        sample_losses.append(token_loss[mask].mean())
        token_kls.append(approximate_kl[mask])
        token_ratios.append(ratio[mask])
        active_token_count += active
    loss = torch.stack(sample_losses).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("policy loss is non-finite")
    return PolicyLossResult(
        loss=loss,
        approximate_kl=torch.cat(token_kls).mean().detach(),
        active_token_count=active_token_count,
        ratio_mean=torch.cat(token_ratios).mean().detach(),
    )


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GRPO") from exc


class GRPOObjective:
    def __init__(self, spec: GRPOSpec):
        if not spec.enabled:
            raise ValueError("GRPOObjective requires an enabled specification")
        self.spec = spec

    @property
    def capabilities(self) -> ObjectiveCapabilities:
        return ObjectiveCapabilities(
            token_scope=self.spec.token_scope,
            required_rollouts=self.spec.group_size,
            behavior_logprobs=True,
            rewards=True,
        )

    def compute(self, batch: ObjectiveBatch) -> ObjectiveResult:
        current = batch.policy_scores.get("current")
        behavior = batch.behavior_policy_scores.get("behavior")
        masks = batch.policy_scores.get("masks")
        if current is None or behavior is None or masks is None:
            raise ValueError("GRPO batch requires current, behavior, and mask sequences")
        result = grpo_policy_loss(
            current_logprobs=current,
            behavior_logprobs=behavior,
            advantages=tuple(batch.advantages.values()),
            masks=masks,
            clip_epsilon=self.spec.clip_epsilon,
            kl_coefficient=self.spec.kl_coefficient,
        )
        return ObjectiveResult(
            loss=result.loss,
            active_token_count=result.active_token_count,
            diagnostics={
                "approximate_kl": float(result.approximate_kl),
                "ratio_mean": float(result.ratio_mean),
            },
        )


__all__ = ["GRPOObjective", "PolicyLossResult", "grpo_policy_loss"]
