"""Verify pinned Hub adapters, deterministic partitions, and notebook variables."""

import hashlib
import json

import pytest

from rlm_train.benchmarks import (
    AIME24_SOURCE,
    MATH500_SOURCE,
    JSONLBenchmark,
    prepare_aime24_splits,
    prepare_math500_splits,
    prepare_math_benchmark_splits,
)


def aime_rows(count=30):
    return [
        {
            "id": str(index + 1),
            "problem": f"AIME problem {index + 1}",
            "solution": f"private worked solution {index + 1}",
            "answer": f"{index + 100}",
            "url": f"https://example.test/aime/{index + 1}",
            "year": 2024,
        }
        for index in range(count)
    ]


def math500_rows(count=500):
    return [
        {
            "problem": f"MATH problem {index + 1}",
            "solution": f"private derivation {index + 1}",
            "answer": f"\\boxed{{{index}}}",
            "subject": "Algebra",
            "level": 5,
            "unique_id": f"math-{index + 1:03d}",
        }
        for index in range(count)
    ]


class RecordingLoader:
    def __init__(self, datasets):
        self.datasets = datasets
        self.calls = []

    def __call__(self, repository, config, *, split, revision):
        self.calls.append(
            {
                "repository": repository,
                "config": config,
                "split": split,
                "revision": revision,
            }
        )
        return self.datasets[repository]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_aime24_materialization_is_pinned_private_and_notebook_ready(tmp_path):
    loader = RecordingLoader({AIME24_SOURCE.repository: aime_rows()})
    prepared = prepare_aime24_splits(tmp_path / "aime24", load_dataset_fn=loader)

    train_rows = read_jsonl(prepared.train_path)
    test_rows = read_jsonl(prepared.test_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    variables = prepared.notebook_variables("AIME24")

    assert len(train_rows) == prepared.train_count == 24
    assert len(test_rows) == prepared.test_count == 6
    assert {row["id"] for row in train_rows}.isdisjoint(row["id"] for row in test_rows)
    assert "private worked solution" not in prepared.train_path.read_text(encoding="utf-8")
    assert "private worked solution" not in prepared.test_path.read_text(encoding="utf-8")
    assert set(train_rows[0]) == {"id", "metadata", "prompt", "target"}
    assert loader.calls == [
        {
            "repository": AIME24_SOURCE.repository,
            "config": AIME24_SOURCE.config,
            "split": AIME24_SOURCE.split,
            "revision": AIME24_SOURCE.revision,
        }
    ]
    assert manifest["dataset"]["source"]["revision"] == AIME24_SOURCE.revision
    assert manifest["partition"]["algorithm"] == "sha256-ranked-v1"
    assert manifest["partition"]["train_count"] == 24
    assert manifest["partition"]["test_count"] == 6
    assert variables["AIME24_TRAIN_PATH"] == str(prepared.train_path)
    assert variables["AIME24_TEST_PATH"] == str(prepared.test_path)
    assert variables["AIME24_SOURCE_REVISION"] == AIME24_SOURCE.revision

    training_benchmark = JSONLBenchmark(
        prepared.train_path,
        name="aime24",
        version=AIME24_SOURCE.revision,
        split="train",
    )
    assert len(training_benchmark.problems()) == 24


def test_math500_split_has_exact_count_metadata_and_content_fingerprints(tmp_path):
    loader = RecordingLoader({MATH500_SOURCE.repository: math500_rows()})
    prepared = prepare_math500_splits(tmp_path / "math500", load_dataset_fn=loader)

    train_rows = read_jsonl(prepared.train_path)
    test_rows = read_jsonl(prepared.test_path)
    assert len(train_rows) == 400
    assert len(test_rows) == 100
    assert train_rows[0]["metadata"]["subject"] == "Algebra"
    assert train_rows[0]["metadata"]["level"] == 5
    assert "solution" not in train_rows[0]["metadata"]
    assert (
        prepared.train_fingerprint == hashlib.sha256(prepared.train_path.read_bytes()).hexdigest()
    )
    assert prepared.test_fingerprint == hashlib.sha256(prepared.test_path.read_bytes()).hexdigest()


def test_split_membership_is_independent_of_source_order(tmp_path):
    rows = aime_rows()
    forward = prepare_aime24_splits(
        tmp_path / "forward",
        load_dataset_fn=RecordingLoader({AIME24_SOURCE.repository: rows}),
    )
    reverse = prepare_aime24_splits(
        tmp_path / "reverse",
        load_dataset_fn=RecordingLoader({AIME24_SOURCE.repository: list(reversed(rows))}),
    )

    forward_test_ids = {row["metadata"]["source_id"] for row in read_jsonl(forward.test_path)}
    reverse_test_ids = {row["metadata"]["source_id"] for row in read_jsonl(reverse.test_path)}
    assert forward_test_ids == reverse_test_ids


def test_rerun_is_idempotent_but_rejects_changed_source_content(tmp_path):
    rows = aime_rows()
    loader = RecordingLoader({AIME24_SOURCE.repository: rows})
    first = prepare_aime24_splits(tmp_path / "aime24", load_dataset_fn=loader)
    second = prepare_aime24_splits(tmp_path / "aime24", load_dataset_fn=loader)
    assert first == second

    changed_rows = aime_rows()
    changed_rows[0]["answer"] = "999"
    changed_loader = RecordingLoader({AIME24_SOURCE.repository: changed_rows})
    with pytest.raises(ValueError, match="existing split artifact differs"):
        prepare_aime24_splits(tmp_path / "aime24", load_dataset_fn=changed_loader)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[1].update(id=rows[0]["id"]), "source identity field must be unique"),
        (lambda rows: rows[1].update(problem=rows[0]["problem"]), "source prompts must be unique"),
    ],
)
def test_invalid_source_rows_fail_loudly(tmp_path, mutation, message):
    rows = aime_rows()
    mutation(rows)
    loader = RecordingLoader({AIME24_SOURCE.repository: rows})
    with pytest.raises(ValueError, match=message):
        prepare_aime24_splits(tmp_path / "aime24", load_dataset_fn=loader)


def test_combined_helper_exposes_separate_dataset_results(tmp_path):
    loader = RecordingLoader(
        {
            AIME24_SOURCE.repository: aime_rows(),
            MATH500_SOURCE.repository: math500_rows(),
        }
    )
    prepared = prepare_math_benchmark_splits(tmp_path, load_dataset_fn=loader)

    assert set(prepared) == {"aime24", "math500"}
    assert prepared["aime24"].train_path == (tmp_path / "aime24" / "train.jsonl").resolve()
    assert prepared["math500"].test_path == (tmp_path / "math500" / "test.jsonl").resolve()


def test_notebook_variable_prefix_is_validated(tmp_path):
    loader = RecordingLoader({AIME24_SOURCE.repository: aime_rows()})
    prepared = prepare_aime24_splits(tmp_path / "aime24", load_dataset_fn=loader)
    with pytest.raises(ValueError, match="uppercase identifier"):
        prepared.notebook_variables("24-aime")
