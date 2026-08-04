"""Typed rollout batches and explicit policy/SDPO/Gram objective composition."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rlm_train.judge import QuestionTeacherFeedback
from rlm_train.sdpo import TopKTeacherTarget


@dataclass(frozen=True)
class RolloutSample:
    """Carry exact behavior-policy data for one sampled continuation."""

    trajectory_id: str
    problem_id: str
    group_index: int
    sample_index: int
    prompt: str
    response: str
    prompt_token_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]
    continuation_token_offsets: tuple[tuple[int, int], ...]
    behavior_logprobs: Any
    trainable_token_mask: Any
    reward: float
    advantage: float
    seed: int
    termination_reason: str
    truncated: bool
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Fail on boundary, alignment, reward, or advantage corruption."""
        torch = _torch()
        if not self.trajectory_id or not self.problem_id or not self.prompt_token_ids:
            raise ValueError("rollout identities and prompt tokens must not be empty")
        if not self.continuation_token_ids:
            raise ValueError("rollout continuation must not be empty")
        expected = len(self.continuation_token_ids)
        if len(self.continuation_token_offsets) != expected:
            raise ValueError("continuation token offsets must align with sampled token IDs")
        if any(start < 0 or end < start for start, end in self.continuation_token_offsets):
            raise ValueError("continuation token offsets must be non-negative and ordered")
        if self.behavior_logprobs.shape != (expected,):
            raise ValueError("behavior log-probabilities must align with continuation tokens")
        if self.trainable_token_mask.shape != (expected,):
            raise ValueError("trainable token mask must align with continuation tokens")
        if not torch.isfinite(self.behavior_logprobs).all().item():
            raise ValueError("behavior log-probabilities must be finite")
        if not math.isfinite(self.reward) or not math.isfinite(self.advantage):
            raise ValueError("rewards and advantages must be finite")
        if self.group_index < 0 or self.sample_index < 0 or self.seed < 0:
            raise ValueError("rollout indices and seed must be non-negative")


@dataclass(frozen=True)
class TrainingBatch:
    """Carry replay-complete rollout, feedback, targets, masks, and cache provenance."""

    batch_id: str
    samples: tuple[RolloutSample, ...]
    restricted_feedback: Mapping[str, QuestionTeacherFeedback] = field(default_factory=dict)
    teacher_targets: Mapping[str, TopKTeacherTarget] = field(default_factory=dict)
    sdpo_masks: Mapping[str, Any] = field(default_factory=dict)
    provenance_keys: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Require unique trajectories and aligned optional teacher fields."""
        if not self.batch_id or not self.samples:
            raise ValueError("training batch ID and samples must not be empty")
        for sample in self.samples:
            sample.validate()
        sample_ids = {sample.trajectory_id for sample in self.samples}
        if len(sample_ids) != len(self.samples):
            raise ValueError("training batch trajectory IDs must be unique")
        for name, mapping in (
            ("restricted feedback", self.restricted_feedback),
            ("teacher targets", self.teacher_targets),
            ("SDPO masks", self.sdpo_masks),
        ):
            unknown = set(mapping) - sample_ids
            if unknown:
                raise ValueError(f"{name} references unknown trajectories: {sorted(unknown)!r}")


@dataclass(frozen=True)
class PolicyLossResult:
    """Return policy loss and detached divergence/denominator observations."""

    loss: Any
    approximate_kl: Any
    active_token_count: int
    ratio_mean: Any


@dataclass(frozen=True)
class ObjectiveResult:
    """Return one finite combined loss plus raw and weighted components."""

    total: Any
    raw: dict[str, Any]
    weighted: dict[str, Any]
    coefficients: dict[str, float]
    denominators: dict[str, int]


class ObjectiveComposer:
    """Evaluate only enabled component factories and combine them once."""

    def __init__(self, *, policy_weight: float, sdpo_weight: float, gram_weight: float) -> None:
        coefficients = {
            "policy": policy_weight,
            "sdpo": sdpo_weight,
            "gram": gram_weight,
        }
        if any(value < 0.0 for value in coefficients.values()):
            raise ValueError("objective coefficients must be non-negative")
        if sum(coefficients.values()) <= 0.0:
            raise ValueError("at least one objective coefficient must be positive")
        self.coefficients = coefficients

    def compose(
        self,
        *,
        policy: Callable[[], tuple[Any, int]] | None,
        sdpo: Callable[[], tuple[Any, int]] | None = None,
        gram: Callable[[], tuple[Any, int]] | None = None,
    ) -> ObjectiveResult:
        """Skip disabled model work and attribute non-finite component failures."""
        torch = _torch()
        factories = {"policy": policy, "sdpo": sdpo, "gram": gram}
        raw: dict[str, Any] = {}
        weighted: dict[str, Any] = {}
        denominators: dict[str, int] = {}
        reference = None
        for name, coefficient in self.coefficients.items():
            factory = factories[name]
            if coefficient == 0.0:
                continue
            if factory is None:
                raise ValueError(f"enabled {name} objective has no loss factory")
            value, denominator = factory()
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"{name} loss must be a scalar PyTorch tensor")
            if not torch.isfinite(value).item():
                raise FloatingPointError(
                    f"non-finite {name} loss (coefficient={coefficient}, denominator={denominator})"
                )
            if denominator <= 0:
                raise ValueError(f"{name} objective denominator must be positive")
            raw[name] = value
            weighted[name] = value * coefficient
            denominators[name] = denominator
            reference = value
        if reference is None:
            raise RuntimeError("objective composition produced no active loss")
        for name in self.coefficients:
            if name in raw:
                continue
            zero = reference * 0.0
            raw[name] = zero
            weighted[name] = zero
            denominators[name] = 0
        total = torch.stack([weighted[name] for name in ("policy", "sdpo", "gram")]).sum()
        if not torch.isfinite(total).item():
            raise FloatingPointError("combined objective is non-finite")
        return ObjectiveResult(
            total=total,
            raw=raw,
            weighted=weighted,
            coefficients=dict(self.coefficients),
            denominators=denominators,
        )


def group_relative_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Normalize rewards within a group; zero-variance groups receive exact zeros."""
    if not rewards:
        raise ValueError("advantage calculation requires at least one reward")
    if any(not math.isfinite(value) for value in rewards):
        raise ValueError("group rewards must be finite")
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    if variance == 0.0:
        return (0.0,) * len(rewards)
    scale = math.sqrt(variance)
    return tuple((value - mean) / scale for value in rewards)


def grpo_policy_loss(
    *,
    current_logprobs: Sequence[Any],
    behavior_logprobs: Sequence[Any],
    advantages: Sequence[float],
    masks: Sequence[Any],
    clip_epsilon: float,
    kl_coefficient: float = 0.0,
) -> PolicyLossResult:
    """Compute the clipped grouped policy objective over continuation tokens only."""
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
            raise ValueError("each rollout requires at least one trainable continuation token")
        if not torch.isfinite(current).all().item() or not torch.isfinite(behavior).all().item():
            raise FloatingPointError("policy log-probabilities must be finite")
        log_ratio = current - behavior.detach().to(current.device)
        ratio = torch.exp(log_ratio)
        advantage_tensor = torch.as_tensor(float(advantage), device=current.device)
        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_tensor
        token_objective = torch.minimum(unclipped, clipped)
        approximate_kl = torch.exp(log_ratio) - 1.0 - log_ratio
        token_loss = -token_objective + kl_coefficient * approximate_kl
        sample_losses.append(token_loss[mask].mean())
        token_kls.append(approximate_kl[mask])
        token_ratios.append(ratio[mask])
        active_token_count += active
    loss = torch.stack(sample_losses).mean()
    concatenated_kl = torch.cat(token_kls)
    concatenated_ratio = torch.cat(token_ratios)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("policy loss is non-finite")
    return PolicyLossResult(
        loss=loss,
        approximate_kl=concatenated_kl.mean().detach(),
        active_token_count=active_token_count,
        ratio_mean=concatenated_ratio.mean().detach(),
    )


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for local training objectives") from exc


__all__ = [
    "ObjectiveComposer",
    "ObjectiveResult",
    "PolicyLossResult",
    "RolloutSample",
    "TrainingBatch",
    "grpo_policy_loss",
    "group_relative_advantages",
]
