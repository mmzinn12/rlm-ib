"""Prepare deterministic local JSONL splits from pinned Hugging Face datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_SCHEMA_VERSION = 1
SPLIT_ALGORITHM = "sha256-ranked-v1"


@dataclass(frozen=True)
class HubDatasetSource:
    """Identify one immutable split of a Hugging Face dataset repository."""

    repository: str
    revision: str
    config: str
    split: str

    def __post_init__(self) -> None:
        """Reject source identities that cannot provide reproducible provenance."""
        values = (self.repository, self.revision, self.config, self.split)
        if any(not value.strip() for value in values):
            raise ValueError("Hugging Face source fields must not be blank")


@dataclass(frozen=True)
class HubDatasetSplitSpec:
    """Map one Hub dataset schema into private-target benchmark JSONL files."""

    name: str
    source: HubDatasetSource
    identity_field: str
    prompt_field: str
    target_field: str
    metadata_fields: tuple[tuple[str, str], ...]
    id_prefix: str
    test_count: int
    salt: str

    def __post_init__(self) -> None:
        """Validate deterministic split and field-mapping choices."""
        required = (
            self.name,
            self.identity_field,
            self.prompt_field,
            self.target_field,
            self.id_prefix,
            self.salt,
        )
        if any(not value.strip() for value in required):
            raise ValueError("split specification fields must not be blank")
        if self.test_count <= 0:
            raise ValueError("test_count must be positive")
        if any(not source.strip() or not output.strip() for source, output in self.metadata_fields):
            raise ValueError("metadata field mappings must not be blank")
        output_metadata_fields = [output for _, output in self.metadata_fields]
        if len(output_metadata_fields) != len(set(output_metadata_fields)):
            raise ValueError("output metadata field names must be unique")
        reserved_metadata_fields = {"source_id", "source_repository", "source_revision"}
        if reserved_metadata_fields.intersection(output_metadata_fields):
            raise ValueError("output metadata fields cannot replace source provenance")


@dataclass(frozen=True)
class PreparedDatasetSplits:
    """Expose durable split artifacts and notebook-friendly scalar variables."""

    name: str
    source: HubDatasetSource
    train_path: Path
    test_path: Path
    manifest_path: Path
    train_count: int
    test_count: int
    train_fingerprint: str
    test_fingerprint: str

    def notebook_variables(self, prefix: str) -> dict[str, str | int]:
        """Return explicit path, count, fingerprint, and source variables for a notebook."""
        normalized = prefix.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized):
            raise ValueError("notebook variable prefix must be a valid uppercase identifier")
        return {
            f"{normalized}_TRAIN_PATH": str(self.train_path),
            f"{normalized}_TEST_PATH": str(self.test_path),
            f"{normalized}_MANIFEST_PATH": str(self.manifest_path),
            f"{normalized}_TRAIN_COUNT": self.train_count,
            f"{normalized}_TEST_COUNT": self.test_count,
            f"{normalized}_TRAIN_FINGERPRINT": self.train_fingerprint,
            f"{normalized}_TEST_FINGERPRINT": self.test_fingerprint,
            f"{normalized}_SOURCE_REPOSITORY": self.source.repository,
            f"{normalized}_SOURCE_REVISION": self.source.revision,
        }


AIME24_SOURCE = HubDatasetSource(
    repository="HuggingFaceH4/aime_2024",
    revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    config="default",
    split="train",
)
MATH500_SOURCE = HubDatasetSource(
    repository="HuggingFaceH4/MATH-500",
    revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    config="default",
    split="test",
)

AIME24_SPLIT_SPEC = HubDatasetSplitSpec(
    name="aime24",
    source=AIME24_SOURCE,
    identity_field="id",
    prompt_field="problem",
    target_field="answer",
    metadata_fields=(("year", "year"), ("url", "url")),
    id_prefix="aime24-",
    test_count=6,
    salt="rlm-aime24-v1",
)
MATH500_SPLIT_SPEC = HubDatasetSplitSpec(
    name="math500",
    source=MATH500_SOURCE,
    identity_field="unique_id",
    prompt_field="problem",
    target_field="answer",
    metadata_fields=(("subject", "subject"), ("level", "level")),
    id_prefix="math500-",
    test_count=100,
    salt="rlm-math500-v1",
)

LoadDataset = Callable[..., Iterable[Mapping[str, Any]]]


def prepare_hub_dataset_splits(
    specification: HubDatasetSplitSpec,
    output_directory: str | Path,
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> PreparedDatasetSplits:
    """Download a pinned Hub snapshot and materialize deterministic train/test JSONL."""
    loader = load_dataset_fn or _hugging_face_loader()
    loaded = loader(
        specification.source.repository,
        specification.source.config,
        split=specification.source.split,
        revision=specification.source.revision,
    )
    source_rows = tuple(loaded)
    train_rows, test_rows, train_source_ids, test_source_ids = deterministic_partition(
        source_rows,
        specification,
    )

    directory = Path(output_directory).expanduser().resolve()
    train_path = directory / "train.jsonl"
    test_path = directory / "test.jsonl"
    manifest_path = directory / "manifest.json"
    train_payload = _jsonl_payload(train_rows)
    test_payload = _jsonl_payload(test_rows)
    train_fingerprint = _fingerprint(train_payload)
    test_fingerprint = _fingerprint(test_payload)
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset": {
            "name": specification.name,
            "source": asdict(specification.source),
            "mapping": {
                "identity_field": specification.identity_field,
                "prompt_field": specification.prompt_field,
                "target_field": specification.target_field,
                "metadata_fields": [list(item) for item in specification.metadata_fields],
                "id_prefix": specification.id_prefix,
            },
        },
        "partition": {
            "algorithm": SPLIT_ALGORITHM,
            "salt": specification.salt,
            "source_count": len(source_rows),
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_source_ids": train_source_ids,
            "test_source_ids": test_source_ids,
        },
        "artifacts": {
            "train": {
                "filename": train_path.name,
                "sha256": train_fingerprint,
            },
            "test": {
                "filename": test_path.name,
                "sha256": test_fingerprint,
            },
        },
    }
    manifest_payload = _json_payload(manifest)
    _materialize_snapshot(
        (
            (train_path, train_payload),
            (test_path, test_payload),
            (manifest_path, manifest_payload),
        )
    )

    return PreparedDatasetSplits(
        name=specification.name,
        source=specification.source,
        train_path=train_path,
        test_path=test_path,
        manifest_path=manifest_path,
        train_count=len(train_rows),
        test_count=len(test_rows),
        train_fingerprint=train_fingerprint,
        test_fingerprint=test_fingerprint,
    )


def deterministic_partition(
    source_rows: Iterable[Mapping[str, Any]],
    specification: HubDatasetSplitSpec,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Select an exact test count by salted SHA-256 rank while preserving source order."""
    prepared: list[tuple[str, dict[str, Any]]] = []
    source_ids: set[str] = set()
    prompts: set[str] = set()
    for index, source_row in enumerate(source_rows):
        if not isinstance(source_row, Mapping):
            raise ValueError(f"source row {index} must be a mapping")
        source_id = _required_text(source_row, specification.identity_field, index)
        if source_id in source_ids:
            raise ValueError(f"source identity field must be unique: {source_id!r}")
        source_ids.add(source_id)
        prompt = _required_text(source_row, specification.prompt_field, index)
        normalized_prompt = " ".join(prompt.split()).casefold()
        if normalized_prompt in prompts:
            raise ValueError(f"source prompts must be unique; duplicate at row {index}")
        prompts.add(normalized_prompt)
        target = _required_text(source_row, specification.target_field, index)
        metadata: dict[str, Any] = {
            "source_id": source_id,
            "source_repository": specification.source.repository,
            "source_revision": specification.source.revision,
        }
        for source_field, output_field in specification.metadata_fields:
            if source_field not in source_row:
                raise ValueError(f"source row {index} is missing field {source_field!r}")
            metadata[output_field] = source_row[source_field]
        _validate_json(metadata, description=f"metadata for source row {index}")
        prepared.append(
            (
                source_id,
                {
                    "id": f"{specification.id_prefix}{source_id}",
                    "prompt": prompt,
                    "target": target,
                    "metadata": metadata,
                },
            )
        )
    if not prepared:
        raise ValueError("source dataset must contain at least one row")
    if specification.test_count >= len(prepared):
        raise ValueError("test_count must leave at least one training row")

    ranked = sorted(
        prepared,
        key=lambda item: hashlib.sha256(f"{specification.salt}\0{item[0]}".encode()).digest(),
    )
    test_source_id_set = {source_id for source_id, _ in ranked[: specification.test_count]}
    train_rows = tuple(row for source_id, row in prepared if source_id not in test_source_id_set)
    test_rows = tuple(row for source_id, row in prepared if source_id in test_source_id_set)
    train_source_ids = tuple(
        source_id for source_id, _ in prepared if source_id not in test_source_id_set
    )
    test_source_ids = tuple(
        source_id for source_id, _ in prepared if source_id in test_source_id_set
    )
    return train_rows, test_rows, train_source_ids, test_source_ids


def prepare_aime24_splits(
    output_directory: str | Path,
    *,
    test_count: int = AIME24_SPLIT_SPEC.test_count,
    salt: str = AIME24_SPLIT_SPEC.salt,
    load_dataset_fn: LoadDataset | None = None,
) -> PreparedDatasetSplits:
    """Prepare the pinned AIME 2024 snapshot with a 24/6 split by default."""
    specification = replace(AIME24_SPLIT_SPEC, test_count=test_count, salt=salt)
    return prepare_hub_dataset_splits(
        specification,
        output_directory,
        load_dataset_fn=load_dataset_fn,
    )


def prepare_math500_splits(
    output_directory: str | Path,
    *,
    test_count: int = MATH500_SPLIT_SPEC.test_count,
    salt: str = MATH500_SPLIT_SPEC.salt,
    load_dataset_fn: LoadDataset | None = None,
) -> PreparedDatasetSplits:
    """Prepare the pinned MATH-500 snapshot with a 400/100 split by default."""
    specification = replace(MATH500_SPLIT_SPEC, test_count=test_count, salt=salt)
    return prepare_hub_dataset_splits(
        specification,
        output_directory,
        load_dataset_fn=load_dataset_fn,
    )


def prepare_math_benchmark_splits(
    output_root: str | Path,
    *,
    load_dataset_fn: LoadDataset | None = None,
) -> dict[str, PreparedDatasetSplits]:
    """Prepare both built-in datasets beneath one root for notebook use."""
    root = Path(output_root)
    return {
        "aime24": prepare_aime24_splits(root / "aime24", load_dataset_fn=load_dataset_fn),
        "math500": prepare_math500_splits(root / "math500", load_dataset_fn=load_dataset_fn),
    }


def main() -> None:
    """Materialize one or both built-in splits and print notebook variables as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("aime24", "math500", "all"))
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.dataset == "aime24":
        prepared = {"aime24": prepare_aime24_splits(arguments.output_root / "aime24")}
    elif arguments.dataset == "math500":
        prepared = {"math500": prepare_math500_splits(arguments.output_root / "math500")}
    else:
        prepared = prepare_math_benchmark_splits(arguments.output_root)
    variables: dict[str, str | int] = {}
    for name, splits in prepared.items():
        variables.update(splits.notebook_variables(name))
    print(json.dumps(variables, indent=2, sort_keys=True))


def _hugging_face_loader() -> LoadDataset:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face dataset preparation requires the 'hub-datasets' extra: "
            "pip install -e './training[hub-datasets]'"
        ) from exc
    return load_dataset


def _required_text(row: Mapping[str, Any], field: str, row_index: int) -> str:
    if field not in row:
        raise ValueError(f"source row {row_index} is missing field {field!r}")
    raw_value = row[field]
    if raw_value is None:
        raise ValueError(f"source row {row_index} has blank field {field!r}")
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"source row {row_index} has blank field {field!r}")
    return value


def _validate_json(value: Any, *, description: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be JSON serializable") from exc


def _jsonl_payload(rows: tuple[dict[str, Any], ...]) -> bytes:
    lines = [
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _json_payload(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _materialize_snapshot(artifacts: tuple[tuple[Path, bytes], ...]) -> None:
    for path, payload in artifacts:
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"existing split artifact differs from pinned source: {path}")
    for path, payload in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)


__all__ = [
    "AIME24_SOURCE",
    "AIME24_SPLIT_SPEC",
    "MATH500_SOURCE",
    "MATH500_SPLIT_SPEC",
    "HubDatasetSource",
    "HubDatasetSplitSpec",
    "PreparedDatasetSplits",
    "deterministic_partition",
    "prepare_aime24_splits",
    "prepare_hub_dataset_splits",
    "prepare_math500_splits",
    "prepare_math_benchmark_splits",
]
