"""AdamW optimizer construction from the runtime specification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlm_train.spec.run import RuntimeSpec


def build_optimizer(parameters: Iterable[Any], runtime: RuntimeSpec) -> Any:
    """Create the AdamW optimizer over the given parameters at the runtime learning rate.

    Args:
        parameters: Trainable parameters to optimize.
        runtime: Runtime settings supplying the learning rate.

    Returns:
        A ``torch.optim.AdamW`` instance over the materialized parameters.

    Raises:
        ValueError: If no trainable parameters are provided.
    """
    torch = __import__("torch")
    materialized = list(parameters)
    if not materialized:
        raise ValueError("optimizer requires at least one trainable parameter")
    return torch.optim.AdamW(materialized, lr=runtime.learning_rate)


__all__ = ["build_optimizer"]
