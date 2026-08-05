"""Resolve an optional Drive artifact destination."""

from pathlib import Path


def artifact_directory(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


__all__ = ["artifact_directory"]
