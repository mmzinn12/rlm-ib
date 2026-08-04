"""Expose runtime training types lazily so pure objective modules stay importable."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "RLMTrainEnv",
    "RLMTrainRubric",
    "ReplBackend",
    "ExecResult",
    "SubprocessReplBackend",
    "SubLLMProxy",
    "ClientHandle",
]

_EXPORT_MODULES = {
    "RLMTrainEnv": "rlm_train.env",
    "RLMTrainRubric": "rlm_train.rubric",
    "ReplBackend": "rlm_train.repl",
    "ExecResult": "rlm_train.repl",
    "SubprocessReplBackend": "rlm_train.repl",
    "SubLLMProxy": "rlm_train.proxy",
    "ClientHandle": "rlm_train.proxy",
}


def __getattr__(name: str) -> Any:
    """Load Verifiers-dependent runtime modules only when callers request them."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
