"""Load and deterministically replay canonical annotated rollouts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rlm_train.trajectory.schema import AnnotatedRollout


def load_annotated_rollout(path: str | Path) -> AnnotatedRollout:
    return AnnotatedRollout.model_validate_json(Path(path).read_text(encoding="utf-8"))


def replay_annotated_events(rollout: AnnotatedRollout) -> tuple[dict[str, object], ...]:
    """Return the immutable ordered event stream after validating the record."""
    AnnotatedRollout.model_validate(rollout.model_dump(mode="python"))
    return tuple(dict(event) for event in rollout.execution.events)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_path")
    args = parser.parse_args(argv)
    rollout = load_annotated_rollout(args.rollout_path)
    print(json.dumps(replay_annotated_events(rollout), sort_keys=True, indent=2))
    return 0


__all__ = ["load_annotated_rollout", "replay_annotated_events"]
