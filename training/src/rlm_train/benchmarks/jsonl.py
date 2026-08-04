"""Implement a deterministic, target-private generic JSONL benchmark adapter."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlm_train.benchmarks.types import (
    BenchmarkIdentity,
    BenchmarkRole,
    ExtractedAnswer,
    ExtractionStatus,
    Problem,
    Score,
)


class JSONLBenchmark:
    """Read public prompts and verifier targets from a local JSONL snapshot.

    Each row must contain ``id``, ``prompt``, and ``target``. ``metadata`` is optional.
    Target values never enter formatted prompts or persisted evaluation records.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        name: str,
        version: str,
        split: str,
        role: BenchmarkRole = BenchmarkRole.DEVELOPMENT,
        answer_pattern: str | None = None,
        case_sensitive: bool = True,
    ) -> None:
        """Load, validate, fingerprint, and freeze one JSONL dataset snapshot."""
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"benchmark JSONL does not exist: {self.path}")
        if answer_pattern is not None and not answer_pattern.strip():
            raise ValueError("answer_pattern must not be blank")
        self.answer_pattern = answer_pattern
        self.case_sensitive = case_sensitive
        self._answer_regex = re.compile(answer_pattern, re.MULTILINE) if answer_pattern else None
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid benchmark JSON on line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"benchmark line {line_number} must contain an object")
            unknown = set(row) - {"id", "prompt", "target", "metadata"}
            if unknown:
                raise ValueError(
                    f"benchmark line {line_number} has unknown fields: {sorted(unknown)!r}"
                )
            missing = {"id", "prompt", "target"} - set(row)
            if missing:
                raise ValueError(
                    f"benchmark line {line_number} is missing fields: {sorted(missing)!r}"
                )
            rows.append(row)
        if not rows:
            raise ValueError("benchmark JSONL must contain at least one problem")
        problem_ids = [str(row["id"]) for row in rows]
        if any(not problem_id.strip() for problem_id in problem_ids):
            raise ValueError("benchmark problem IDs must not be blank")
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("benchmark problem IDs must be unique")
        canonical = json.dumps(
            {
                "rows": rows,
                "answer_pattern": answer_pattern,
                "case_sensitive": case_sensitive,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self._identity = BenchmarkIdentity(
            name=name,
            version=version,
            split=split,
            source_fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            role=role,
        )
        self._problems = tuple(
            Problem(
                problem_id=str(row["id"]),
                public_prompt=str(row["prompt"]),
                metadata=dict(row.get("metadata") or {}),
                target_data=row["target"],
            )
            for row in rows
        )

    @property
    def identity(self) -> BenchmarkIdentity:
        """Return the content-addressed benchmark identity."""
        return self._identity

    def problems(self) -> Sequence[Problem]:
        """Return immutable problems in source order."""
        return self._problems

    def format_prompt(self, problem: Problem) -> str:
        """Expose only the public prompt, never verifier-owned target data."""
        if problem not in self._problems:
            raise ValueError("problem does not belong to this benchmark snapshot")
        return problem.public_prompt

    def extract_answer(self, response: str) -> ExtractedAnswer:
        """Extract the named/first regex group or the stripped complete response."""
        if not response.strip():
            return ExtractedAnswer(status=ExtractionStatus.MISSING)
        if self._answer_regex is None:
            value = response.strip()
        else:
            matches = list(self._answer_regex.finditer(response))
            if not matches:
                return ExtractedAnswer(status=ExtractionStatus.MALFORMED)
            match = matches[-1]
            if "answer" in match.groupdict():
                value = match.group("answer")
            elif match.lastindex:
                value = match.group(1)
            else:
                value = match.group(0)
            value = value.strip()
        if not value:
            return ExtractedAnswer(status=ExtractionStatus.MALFORMED)
        return ExtractedAnswer(
            normalized_answer=self._normalize(value),
            status=ExtractionStatus.EXTRACTED,
        )

    def score(self, problem: Problem, answer: ExtractedAnswer) -> Score:
        """Apply exact normalized verification with structured failure reasons."""
        if problem not in self._problems:
            raise ValueError("problem does not belong to this benchmark snapshot")
        if answer.status is ExtractionStatus.MISSING:
            return Score(reward=0.0, correct=False, failure_reason="missing_answer")
        if answer.status is ExtractionStatus.MALFORMED:
            return Score(reward=0.0, correct=False, failure_reason="malformed_answer")
        target = self._normalize(str(problem.target_data))
        correct = answer.normalized_answer == target
        return Score(
            reward=1.0 if correct else 0.0,
            correct=correct,
            failure_reason=None if correct else "incorrect_answer",
        )

    def _normalize(self, value: str) -> str:
        """Normalize surrounding whitespace and optional case only."""
        value = value.strip()
        return value if self.case_sensitive else value.casefold()


def find_prompt_overlaps(
    benchmark: JSONLBenchmark,
    training_inputs: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """Return benchmark IDs whose normalized prompts appear in training inputs."""
    indexed: dict[str, list[int]] = {}
    for index, value in enumerate(training_inputs):
        normalized = " ".join(value.split()).casefold()
        indexed.setdefault(normalized, []).append(index)
    overlaps: dict[str, tuple[int, ...]] = {}
    for problem in benchmark.problems():
        normalized = " ".join(problem.public_prompt.split()).casefold()
        if normalized in indexed:
            overlaps[problem.problem_id] = tuple(indexed[normalized])
    return overlaps


__all__ = ["JSONLBenchmark", "find_prompt_overlaps"]
