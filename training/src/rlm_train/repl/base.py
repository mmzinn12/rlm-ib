"""Define the training REPL backend contract and normalized execution result.

Purpose:
    Decouple the rollout environment from subprocess or future execution backends.
Implementation:
    ``ExecResult`` normalizes worker output and ``ReplBackend`` defines asynchronous
    lifecycle, context loading, traced execution, and bootstrap operations.
Inputs:
    Proxy connection details, context payloads, code, and optional trace context.
Outputs:
    Backend state changes and normalized ``ExecResult`` objects.
Example:
    ``result = await backend.execute(code, trace_context={"parent_node_id": node_id})``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecResult:
    """Represent normalized output from one persistent-REPL execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error and execution traceback, when present.
        final_answer: Answer surfaced by the reserved ``answer`` dictionary.
        execution_time: Worker-side elapsed execution time in seconds.
        locals_keys: Serializable user-local variable names after execution.
        trace_call_count: Number of single or batched call sites executed in the block.
    """

    stdout: str = ""
    stderr: str = ""
    final_answer: str | None = None
    execution_time: float = 0.0
    locals_keys: list[str] = field(default_factory=list)
    trace_call_count: int = 0


class ReplBackend(ABC):
    """Specify the asynchronous lifecycle required by ``RLMTrainEnv``."""

    @abstractmethod
    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None:
        """Start backend resources for one rollout and configure subcall routing."""
        ...

    @abstractmethod
    async def load_context(self, payload: Any, index: int | None = None) -> int:
        """Load a context payload and return the numeric context-variable index."""
        ...

    @abstractmethod
    async def execute(self, code: str, trace_context: dict[str, Any] | None = None) -> ExecResult:
        """Execute code with optional parent/call metadata and return normalized output.

        Args:
            code: Python source generated inside a fenced REPL block.
            trace_context: Optional parent node, block index, and call-order offset to
                propagate through worker subcalls.

        Returns:
            Captured execution output and the number of dynamic subcall sites.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Release all backend resources and return ``None``."""
        ...

    async def bootstrap(self, code: str) -> None:
        """Execute optional setup code before rollout iterations begin.

        Args:
            code: Setup source; empty input is a no-op.
        """
        if not code:
            return
        await self.execute(code)
