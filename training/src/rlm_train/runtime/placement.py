"""Runtime placement decision independent of notebook detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Placement:
    device: str
    precision: str


def resolve_placement(*, device: str, precision: str) -> Placement:
    if device == "auto":
        try:
            torch = __import__("torch")
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cpu" and precision == "fp16":
        raise ValueError("fp16 single-device training requires CUDA")
    return Placement(device=device, precision=precision)


__all__ = ["Placement", "resolve_placement"]
