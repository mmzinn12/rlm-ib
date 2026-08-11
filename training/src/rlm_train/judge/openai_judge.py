"""OpenAI-compatible judge for structured feedback assessments."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from rlm_train.feedback.feedback_records import ScopedAssessment
from rlm_train.feedback.feedback_views import FeedbackView
from rlm_train.judge.cache import JudgeCache, make_judge_view_cache_key
from rlm_train.judge.prompt import create_judge_instructions, render_feedback_view
from rlm_train.judge.response_formats import (
    CategoricalJudgeAssessment,
    FullJudgeAssessment,
)
from rlm_train.settings.judge import JudgeMode, JudgeSettings


class OpenAIJudge:
    """Assess an authorized feedback view through an OpenAI-compatible Responses API."""

    def __init__(
        self,
        settings: JudgeSettings,
        *,
        client: Any | None = None,
        cache: JudgeCache | None = None,
    ) -> None:
        if settings.provider != "openai":
            raise ValueError("OpenAIJudge requires provider='openai'")
        if client is None:
            api_key = os.environ.get(settings.api_key_environment, "")
            if not api_key.strip():
                raise RuntimeError(
                    f"required judge secret {settings.api_key_environment!r} is absent"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "the openai package is required for an OpenAI-compatible judge"
                ) from exc
            arguments: dict[str, Any] = {"api_key": api_key}
            if settings.base_url is not None:
                arguments["base_url"] = str(settings.base_url)
            client = OpenAI(**arguments)
        self.settings = settings
        self.client = client
        self.cache = cache

    @property
    def provider_identity(self) -> str:
        return f"openai:{self.settings.mode.value}"

    @property
    def cache_prompt_version(self) -> str:
        return f"{self.settings.prompt_version}:task-relevance-v2:{self.settings.schema_name}"

    def assess(self, view: FeedbackView) -> ScopedAssessment:
        cache_key = make_judge_view_cache_key(
            provider=self.provider_identity,
            model_revision=self.settings.model_revision,
            prompt_version=self.cache_prompt_version,
            view_fingerprint=view.fingerprint,
        )
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        assessment_model = (
            CategoricalJudgeAssessment
            if self.settings.mode is JudgeMode.CATEGORICAL
            else FullJudgeAssessment
        )
        instructions, view_payload = render_feedback_view(
            view,
            instructions=self.instructions,
        )
        previous_error: str | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            input_payload = json.dumps(
                {
                    "judge_view": json.loads(view_payload),
                    "attempt": attempt,
                    "previous_validation_error": previous_error,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            response = self.client.responses.create(
                model=self.settings.model,
                instructions=instructions,
                input=input_payload,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": self.settings.schema_name,
                        "schema": assessment_model.model_json_schema(),
                        "strict": True,
                    }
                },
            )
            output_text = getattr(response, "output_text", None)
            try:
                if not isinstance(output_text, str) or not output_text.strip():
                    raise ValueError("judge returned no structured output text")
                parsed = assessment_model.model_validate_json(output_text)
            except (TypeError, ValueError) as exc:
                previous_error = str(exc)
                last_error = exc
                continue
            content = parsed.normalized_content()
            assessment = self.create_assessment(
                view,
                content=content,
                cache_key=cache_key,
            )
            if self.cache is not None:
                self.cache.put(cache_key, assessment)
            return assessment
        raise RuntimeError(
            f"{self.settings.mode.value} judge failed validation after "
            f"{self.settings.max_attempts} attempts"
        ) from last_error

    @property
    def instructions(self) -> str:
        if self.settings.mode is JudgeMode.CATEGORICAL:
            contract = (
                "Return categorical labels only. Use exactly the enum values in the schema. "
                "Do not invent numeric rating scales. Assess the focal helper question by the "
                "information revealed in its response relative to the evidence available before "
                "the call. Apply the original-task relevance gate before assigning any label."
            )
        else:
            contract = (
                "Return the full numeric assessment. information_significance must be in "
                "[-1, 1]. novelty, uncertainty_reduction, and evidence_quality must each be "
                "in [0, 1]. Do not use a 1-to-5 or 1-to-10 rating scale."
            )
        rubric = (
            " Also fill the rubric with rich qualitative feedback: what information the question "
            "revealed, what was missing, whether it was redundant or misleading, why it mattered, "
            "and concrete guidance for a sharper question. The rubric is the primary output; the "
            "labels/scores are secondary metadata."
        )
        return create_judge_instructions(contract + rubric)

    def create_assessment(
        self,
        view: FeedbackView,
        *,
        content: dict[str, object],
        cache_key: str,
    ) -> ScopedAssessment:
        identity = json.dumps(
            {"cache_key": cache_key, "content": content},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return ScopedAssessment(
            assessment_id=hashlib.sha256(identity.encode()).hexdigest(),
            scope=view.scope,
            focal_node_ids=view.focal_node_ids,
            focal_edge_ids=view.focal_edge_ids,
            evidence_node_ids=view.evidence_node_ids,
            evidence_event_ids=view.evidence_event_ids,
            judge_view_fingerprint=view.fingerprint,
            content=content,
            visibility=view.visibility,
            future_public_events_visible=view.downstream_depth > 0,
            final_answer_visible=view.final_answer_included,
            reference_answer_visible=view.verifier_reference_included,
            allowed_objectives=view.allowed_objectives,
            allowed_token_scopes=view.allowed_token_scopes,
            provider=self.provider_identity,
            model_revision=self.settings.model_revision,
            prompt_version=self.cache_prompt_version,
            cache_key=cache_key,
        )


__all__ = ["OpenAIJudge"]
