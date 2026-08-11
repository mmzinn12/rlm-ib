"""Create the optimizer that updates student parameters."""

from rlm_train.engine.optimizer import build_optimizer

create_optimizer = build_optimizer

__all__ = ["create_optimizer"]
