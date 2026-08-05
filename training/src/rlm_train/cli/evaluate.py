"""CLI for whole-recursive-policy evaluation."""

from __future__ import annotations

import argparse

from rlm_train.api import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_spec")
    arguments = parser.parse_args()
    evaluate(arguments.run_spec)


if __name__ == "__main__":
    main()
