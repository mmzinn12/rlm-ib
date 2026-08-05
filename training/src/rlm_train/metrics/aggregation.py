"""Pure aggregation of scalar metric observations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from rlm_train.metrics.schema import MetricObservation


def mean_by_name(observations: Iterable[MetricObservation]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        values[observation.name].append(observation.value)
    return {name: sum(items) / len(items) for name, items in sorted(values.items())}


__all__ = ["mean_by_name"]
