"""Regression tests for package import boundaries."""

import subprocess
import sys


def test_prepare_batch_imports_in_fresh_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rlm_train.training.prepare_batch import LossResult, TrainingBatch",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
