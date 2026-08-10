"""Learning-rate scheduler construction isolated from run configuration parsing."""

from __future__ import annotations

from typing import Any

from rlm_train.spec.run import RuntimeSpec


def build_scheduler(optimizer: Any, *, kind: str, warmup_steps: int, total_steps: int) -> Any:
    transformers = __import__("transformers")
    if kind not in {"constant", "linear", "cosine"}:
        raise ValueError(f"unsupported scheduler {kind!r}")
    name = "constant_with_warmup" if kind == "constant" else kind
    return transformers.get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def build_training_scheduler(
    optimizer: Any, runtime: RuntimeSpec, *, total_steps: int
) -> Any | None:
    """Return a warmup scheduler when runtime.warmup_steps is set, else None (constant LR)."""
    if runtime.warmup_steps <= 0:
        return None
    return build_scheduler(
        optimizer,
        kind=runtime.scheduler,
        warmup_steps=runtime.warmup_steps,
        total_steps=total_steps,
    )


__all__ = ["build_scheduler", "build_training_scheduler"]
