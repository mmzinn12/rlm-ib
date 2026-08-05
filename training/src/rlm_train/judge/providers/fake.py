"""Deterministic scoped judge used by CPU tests and offline research."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rlm_train.feedback.schema import ScopedAssessment
from rlm_train.judge.cache import make_judge_view_cache_key
from rlm_train.judge.views import JudgeView


class DeterministicFakeJudge:
    def __init__(
        self,
        *,
        content: dict[str, Any] | None = None,
        model_revision: str = "v1",
        prompt_version: str = "v1",
    ) -> None:
        self.content = dict(content or {"quality": "good", "score": 1.0})
        self.model_revision = model_revision
        self.prompt_version = prompt_version

    def assess(self, view: JudgeView) -> ScopedAssessment:
        cache_key = make_judge_view_cache_key(
            provider="fake",
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
            view_fingerprint=view.fingerprint,
        )
        identity = json.dumps(
            {"cache_key": cache_key, "content": self.content},
            sort_keys=True,
            separators=(",", ":"),
        )
        return ScopedAssessment(
            assessment_id=hashlib.sha256(identity.encode()).hexdigest(),
            scope=view.scope,
            focal_node_ids=view.focal_node_ids,
            focal_edge_ids=view.focal_edge_ids,
            evidence_node_ids=view.evidence_node_ids,
            evidence_event_ids=view.evidence_event_ids,
            judge_view_fingerprint=view.fingerprint,
            content=self.content,
            visibility=view.visibility,
            future_public_events_visible=view.downstream_depth > 0,
            final_answer_visible=view.final_answer_included,
            reference_answer_visible=view.verifier_reference_included,
            allowed_objectives=view.allowed_objectives,
            allowed_token_scopes=view.allowed_token_scopes,
            provider="fake",
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
            cache_key=cache_key,
        )


__all__ = ["DeterministicFakeJudge"]
