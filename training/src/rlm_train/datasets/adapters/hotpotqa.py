"""HotpotQA Hub adapter with one fixed, production-shaped record mapping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from rlm_train.datasets.records import DatasetRecord

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


def render_hotpotqa_context(context: Mapping[str, Any]) -> str:
    """Render HotpotQA's parallel title and sentence columns as titled evidence sections."""
    try:
        titles = context["title"]
        sentence_groups = context["sentences"]
    except KeyError as exc:
        raise ValueError("HotpotQA context requires title and sentences columns") from exc
    if (
        not isinstance(titles, Sequence)
        or isinstance(titles, (str, bytes))
        or not isinstance(sentence_groups, Sequence)
        or isinstance(sentence_groups, (str, bytes))
        or len(titles) != len(sentence_groups)
    ):
        raise ValueError("HotpotQA context titles and sentence groups must align")
    sections = []
    for title, sentences in zip(titles, sentence_groups, strict=True):
        if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
            raise ValueError("HotpotQA context sentence group must be a sequence")
        sections.append(f"### {title}\n{''.join(str(sentence) for sentence in sentences)}")
    rendered = "\n\n".join(sections)
    if not rendered.strip():
        raise ValueError("HotpotQA context must be non-empty")
    return rendered


class HotpotQADataset:
    """Stream one HotpotQA split and map its fixed columns into canonical records."""

    def __init__(
        self,
        repository: str,
        *,
        subset: str,
        split: str,
        revision: str | None = None,
        max_records: int | None = None,
        loader: DatasetLoader | None = None,
    ) -> None:
        if not repository.strip() or not subset.strip() or not split.strip():
            raise ValueError("HotpotQA repository, subset, and split must not be blank")
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive")
        self.repository = repository
        self.subset = subset
        self.split = split
        self.revision = revision
        self.max_records = max_records
        self.loader = loader
        self.cached_records: tuple[DatasetRecord, ...] | None = None

    @property
    def identity(self) -> str:
        configuration = json.dumps(
            {
                "repository": self.repository,
                "subset": self.subset,
                "split": self.split,
                "revision": self.revision,
                "max_records": self.max_records,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"hotpotqa:{hashlib.sha256(configuration.encode()).hexdigest()}"

    def records(self) -> tuple[DatasetRecord, ...]:
        if self.cached_records is not None:
            return self.cached_records
        loader = self.loader
        if loader is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("HotpotQA requires the 'hub-datasets' extra") from exc
            loader = load_dataset
        rows = loader(
            self.repository,
            self.subset,
            split=self.split,
            revision=self.revision,
            streaming=True,
        )
        values = []
        for row_number, row in enumerate(rows, start=1):
            values.append(self.convert_row(row, row_number=row_number))
            if self.max_records is not None and len(values) >= self.max_records:
                break
        if not values:
            raise ValueError("HotpotQA split contains no records")
        identifiers = [record.record_id for record in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("HotpotQA record IDs must be unique")
        self.cached_records = tuple(values)
        return self.cached_records

    @staticmethod
    def _validate_row(row, row_number):
        required = {"id", "question", "context", "answer", "type", "level"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"HotpotQA row {row_number} is missing columns {sorted(missing)!r}")
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"HotpotQA row {row_number} question must be non-empty")
        context = row["context"]
        if not isinstance(context, Mapping):
            raise ValueError(f"HotpotQA row {row_number} context must be an object")
        return question, context

    @staticmethod
    def convert_row(row: Mapping[str, Any], *, row_number: int) -> DatasetRecord:
        question, context = HotpotQADataset._validate_row(row, row_number)
        return DatasetRecord(
            record_id=str(row["id"]),
            public_task={
                "question": question,
                "context": render_hotpotqa_context(context),
            },
            verifier_data=row["answer"],
            metadata={"type": row["type"], "level": row["level"]},
        )


__all__ = ["HotpotQADataset", "render_hotpotqa_context"]
