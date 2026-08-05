"""CLI for the canonical public training API."""

from __future__ import annotations

import argparse

from rlm_train.api import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_spec")
    arguments = parser.parse_args()
    train(arguments.run_spec)


if __name__ == "__main__":
    main()
