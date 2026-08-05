"""Observer-only metric collector."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from rlm_train.metrics.schema import MetricObservation


class MetricCollector:
    def __init__(self) -> None:
        self._observations: list[MetricObservation] = []
        self._lock = threading.Lock()

    def record(self, observation: MetricObservation) -> None:
        with self._lock:
            self._observations.append(observation)

    def extend(self, observations: Iterable[MetricObservation]) -> None:
        for observation in observations:
            self.record(observation)

    @property
    def observations(self) -> tuple[MetricObservation, ...]:
        with self._lock:
            return tuple(self._observations)


__all__ = ["MetricCollector"]
