"""Colab-only preflight; no rollout, objective, or trainer semantics live here."""

from __future__ import annotations

from typing import Any


def validate_colab_device() -> dict[str, Any]:
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab training requires a CUDA runtime")
    return {
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }


__all__ = ["validate_colab_device"]
