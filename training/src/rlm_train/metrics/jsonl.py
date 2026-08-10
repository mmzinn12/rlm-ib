"""Append-only JSONL metric sink."""

from __future__ import annotations

import json
from pathlib import Path

from rlm_train.metrics.schema import MetricObservation


class JSONLMetricSink:
    """Append-only sink writing one JSON-serialized metric observation per line.

    The parent directory is created on construction; each ``write`` appends a single sorted-key
    JSON line, so a run's ``metrics.jsonl`` accumulates one row per emitted observation in order.

    Attributes:
        path: Destination JSONL file that observations are appended to.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, observation: MetricObservation) -> None:
        """Append one observation as a JSON line to the sink's file."""
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observation.model_dump(mode="json"), sort_keys=True))
            stream.write("\n")


__all__ = ["JSONLMetricSink"]
