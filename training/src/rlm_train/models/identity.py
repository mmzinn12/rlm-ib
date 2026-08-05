"""Stable model, tokenizer, adapter, policy-owner, and checkpoint identities."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class ComponentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    fingerprint: str | None = None

    @property
    def resolved_fingerprint(self) -> str:
        if self.fingerprint is not None:
            return self.fingerprint
        payload = json.dumps(
            {"component_id": self.component_id, "revision": self.revision},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class PolicyIdentity(ComponentIdentity):
    policy_owner: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    adapter_id: str | None = None


class TokenizerIdentity(ComponentIdentity):
    vocabulary_size: int | None = Field(default=None, gt=0)


__all__ = ["ComponentIdentity", "PolicyIdentity", "TokenizerIdentity"]
