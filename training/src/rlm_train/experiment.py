"""Run-only experiment settings that live alongside, but outside, the canonical RunSpec."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ExperimentSettings(BaseModel):
    """Notebook/CLI knobs not part of RunSpec: scoring, readout, and checkpoint identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scorer: str = Field(default="exact_match", min_length=1)
    checkpoint_id: str = Field(default="latest", min_length=1)
    predictions_filename: str = Field(default="predictions.jsonl", min_length=1)
    render_predictions: bool = True
    max_render_text_chars: int = Field(default=200, gt=0)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentSettings:
        source = Path(path)
        if source.suffix == ".toml":
            with source.open("rb") as stream:
                payload = tomllib.load(stream)
        elif source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("ExperimentSettings must be loaded from .toml or .json")
        return cls.model_validate(payload)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


__all__ = ["ExperimentSettings"]
