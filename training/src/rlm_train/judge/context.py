"""Define an explicit, non-persistent privileged-context channel for judges.

Purpose:
    Allow evaluators to inspect reference evidence that must never enter student
    prompts, trajectory nodes, or question-teacher feedback.
Implementation:
    ``PrivilegedJudgeContext`` stores a canonical JSON payload privately, exposes it
    only through an explicit judge-payload method, and produces a payload-free
    descriptor for caches and trajectory artifacts.
Inputs:
    A source identity, version, JSON-compatible privileged payload, and optional
    non-sensitive metadata.
Outputs:
    Judge-only payloads, stable fingerprints, and safe persistence descriptors.
Example:
    ``context = PrivilegedJudgeContext("answer-key", "v1", {"reference": "..."})``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from rlm.core.trajectory import TrajectoryTree


def canonical_json(value: Any, *, name: str) -> str:
    """Return deterministic JSON or raise a field-specific validation error."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data") from exc


@dataclass(frozen=True)
class PrivilegedContextDescriptor:
    """Persist identity and content hash without persisting privileged content.

    Attributes:
        source_id: Stable evidence-source identifier.
        version: Source snapshot or schema version.
        fingerprint: SHA-256 digest over the source, version, payload, and metadata.
    """

    source_id: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate non-empty identity fields and a lowercase SHA-256 digest."""
        if not self.source_id.strip() or not self.version.strip():
            raise ValueError("privileged context source_id and version must not be blank")
        if len(self.fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.fingerprint
        ):
            raise ValueError("privileged context fingerprint must be a SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        """Serialize the safe descriptor to JSON-compatible primitives."""
        return {
            "source_id": self.source_id,
            "version": self.version,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivilegedContextDescriptor:
        """Reconstruct a descriptor while rejecting missing fields."""
        return cls(
            source_id=str(data["source_id"]),
            version=str(data["version"]),
            fingerprint=str(data["fingerprint"]),
        )


class PrivilegedJudgeContext:
    """Hold judge-only evidence without exposing it through representation or storage.

    The payload is canonicalized at construction and returned as a fresh value only
    when :meth:`to_judge_payload` is called. ``repr(context)`` and
    :meth:`descriptor` never contain the payload.
    """

    __slots__ = (
        "source_id",
        "version",
        "_payload_json",
        "_metadata_json",
        "_fingerprint",
        "_sealed",
    )

    def __init__(
        self,
        source_id: str,
        version: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Validate and seal one privileged evidence snapshot."""
        object.__setattr__(self, "_sealed", False)
        if not source_id.strip() or not version.strip():
            raise ValueError("privileged context source_id and version must not be blank")
        payload_json = canonical_json(payload, name="privileged context payload")
        metadata_json = canonical_json(
            metadata or {},
            name="privileged context metadata",
        )
        fingerprint_payload = f"{source_id}\0{version}\0{payload_json}\0{metadata_json}".encode()
        self.source_id = source_id
        self.version = version
        self._payload_json = payload_json
        self._metadata_json = metadata_json
        self._fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation after the canonical payload and fingerprint are sealed."""
        if getattr(self, "_sealed", False):
            raise AttributeError("privileged judge context is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        """Return an intentionally payload-free representation."""
        return (
            f"PrivilegedJudgeContext(source_id={self.source_id!r}, "
            f"version={self.version!r}, fingerprint={self._fingerprint!r})"
        )

    @property
    def fingerprint(self) -> str:
        """Return the stable content digest used by caches and artifacts."""
        return self._fingerprint

    def descriptor(self) -> PrivilegedContextDescriptor:
        """Return a payload-free persistence and cache descriptor."""
        return PrivilegedContextDescriptor(
            source_id=self.source_id,
            version=self.version,
            fingerprint=self.fingerprint,
        )

    def to_judge_payload(self) -> dict[str, Any]:
        """Materialize a fresh payload explicitly for the judge request boundary."""
        return {
            "source_id": self.source_id,
            "version": self.version,
            "payload": json.loads(self._payload_json),
            "metadata": json.loads(self._metadata_json),
        }


class PrivilegedContextProvider(Protocol):
    """Resolve privileged evidence lazily for one task and completed trajectory."""

    async def get_context(
        self,
        *,
        task_id: str,
        trajectory: TrajectoryTree,
    ) -> PrivilegedJudgeContext | None:
        """Return judge-only evidence or ``None`` when no source is configured."""
        ...


__all__ = [
    "PrivilegedContextDescriptor",
    "PrivilegedContextProvider",
    "PrivilegedJudgeContext",
]
