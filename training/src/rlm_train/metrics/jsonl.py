"""Append-only JSONL metric sink."""

from __future__ import annotations

import json
from pathlib import Path

from rlm_train.metrics.schema import MetricObservation


class JSONLMetricSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, observation: MetricObservation) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observation.model_dump(mode="json"), sort_keys=True))
            stream.write("\n")


__all__ = ["JSONLMetricSink"]
