"""Metric sink construction for training runs.

The trainer emits ``MetricObservation`` rows (loss, gradient norm, ...) each optimizer step.
``build_metric_sink`` is the single entry point that points those rows at a ``metrics.jsonl`` file
inside the run's output directory.
"""

from __future__ import annotations

from pathlib import Path

from rlm_train.metrics.jsonl import JSONLMetricSink


def build_metric_sink(
    output_directory: str | Path, *, filename: str = "metrics.jsonl"
) -> JSONLMetricSink:
    """Create the JSONL metric sink for a run.

    Args:
        output_directory: Run output directory the metrics file is written under.
        filename: Name of the JSONL file inside the output directory.

    Returns:
        A ``JSONLMetricSink`` appending observations to ``output_directory/filename``.
    """
    return JSONLMetricSink(Path(output_directory) / filename)


__all__ = ["build_metric_sink"]
