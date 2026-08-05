"""Resolved run and component provenance written before rollout execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_spec_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_spec: dict[str, Any]
    components: dict[str, Any]
    factory_version: str = Field(min_length=1)
    source: dict[str, Any] = Field(default_factory=dict)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


__all__ = ["RunProvenance"]
