"""Rollout execution boundary consumed by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rlm.core.types import RLMChatCompletion

from rlm_train.trajectory.schema import AnnotatedRollout


@dataclass(frozen=True)
class RolloutRequest:
    task_id: str
    public_task: dict[str, Any]
    private_reference: Any | None = None
    mode: str = "training"


@dataclass(frozen=True)
class RolloutResult:
    completion: RLMChatCompletion
    rollout: AnnotatedRollout


class RolloutEngine(Protocol):
    def execute(self, request: RolloutRequest) -> RolloutResult: ...


__all__ = ["RolloutEngine", "RolloutRequest", "RolloutResult"]
