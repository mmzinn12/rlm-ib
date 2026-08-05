"""Safe one-indented-document-per-rollout artifact writer."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rlm_train.trajectory.schema import AnnotatedRollout


class RolloutJSONWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, rollout: AnnotatedRollout) -> Path:
        destination = self.directory / f"{rollout.rollout_id.replace('/', '_')}.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(rollout.canonical_json(indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination


__all__ = ["RolloutJSONWriter"]
