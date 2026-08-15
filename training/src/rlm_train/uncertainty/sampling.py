"""Reproducible direct-answer sampling from the student policy."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from rlm_train.models.protocol import SampledGeneration
from rlm_train.models.transformers_runtime import GenerationConfig, TransformersResponseGenerator
from rlm_train.uncertainty.schema import (
    AnswerSamplingRequest,
    SemanticSample,
    SemanticSampleBatch,
)


def derive_matched_seed(run_seed: int, rollout_id: str, edge_id: str, sample_index: int) -> int:
    if run_seed < 0 or sample_index < 0:
        raise ValueError("seed inputs must be non-negative")
    payload = f"uncertainty-v1\0{run_seed}\0{rollout_id}\0{edge_id}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def matched_seeds(
    run_seed: int, rollout_id: str, edge_id: str, sample_count: int
) -> tuple[int, ...]:
    if sample_count < 2:
        raise ValueError("semantic uncertainty requires at least two samples")
    return tuple(
        derive_matched_seed(run_seed, rollout_id, edge_id, index) for index in range(sample_count)
    )


class TransformersAnswerSampler:
    """Generate exact IDs and rescore them under the same frozen student checkpoint."""

    def __init__(
        self,
        policy: Any,
        *,
        max_prompt_tokens: int = 512,
        use_chat_template: bool = True,
        allow_prompt_truncation: bool = False,
    ) -> None:
        if not hasattr(policy, "generator") or not (
            hasattr(policy, "score_sampled_ids") or hasattr(policy, "score_tokens")
        ):
            raise TypeError("Transformers uncertainty sampling requires a scoreable student policy")
        self.policy = policy
        self.max_prompt_tokens = max_prompt_tokens
        self.use_chat_template = use_chat_template
        self.allow_prompt_truncation = allow_prompt_truncation
        self.cache: dict[str, SemanticSampleBatch] = {}

    def _get_sample(
        self,
        generator: TransformersResponseGenerator,
        request: AnswerSamplingRequest,
        seed: int,
        index: int,
    ) -> SemanticSample:
        """Generate a single semantic sample for the given request and seed."""

        result = generator.generate_tokenized(request.prompt, seed=seed, sample_index=index)
        score = self._score(result)
        if score.logprobs is None:
            raise ValueError("semantic entropy requires behavior token log probabilities")
        values = _finite_logprob_tuple(score.logprobs)
        if len(values) != len(result.continuation_token_ids):
            raise ValueError("behavior log probabilities do not align with generated IDs")
        identity = {
            "request": request.fingerprint,
            "condition": request.condition,
            "sample_index": index,
            "seed": seed,
            "tokens": result.continuation_token_ids,
        }
        sample_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        return SemanticSample(
            sample_id=sample_id,
            answer=result.response.strip(),
            continuation_token_ids=result.continuation_token_ids,
            token_log_probabilities=values,
            sampling_seed=seed,
        )

    def sample(self, request: AnswerSamplingRequest) -> SemanticSampleBatch:
        """Sample multiple semantic samples for the given request."""

        cache_key = hashlib.sha256(
            f"{self._model_identity.resolved_fingerprint}\0{request.fingerprint}".encode()
        ).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        configuration = GenerationConfig(
            prompt_template_version=request.prompt_version,
            system_prompt="Answer the task directly and return only the answer.",
            use_chat_template=self.use_chat_template,
            max_prompt_tokens=self.max_prompt_tokens,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            allow_prompt_truncation=self.allow_prompt_truncation,
        )

        base = self.policy.generator
        generator = TransformersResponseGenerator(
            base.model,
            base.tokenizer,
            configuration,
            model_context_length=base.model_context_length,
        )
        samples = tuple(
            self._get_sample(generator, request, seed, index)
            for index, seed in enumerate(request.seeds)
        )
        batch = SemanticSampleBatch(
            condition=request.condition,
            samples=samples,
            model_identity=self._model_identity.resolved_fingerprint,
            tokenizer_identity=self._tokenizer_identity.resolved_fingerprint,
            sampling_parameters={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_new_tokens,
                "sample_count": request.sample_count,
                "checkpoint_id": self._model_identity.checkpoint_id,
            },
            prompt_fingerprint=hashlib.sha256(request.prompt.encode()).hexdigest(),
            prompt_version=request.prompt_version,
        )
        self.cache[cache_key] = batch
        return batch

    @property
    def _model_identity(self) -> Any:
        return getattr(self.policy, "identity", getattr(self.policy, "model_info", None))

    @property
    def _tokenizer_identity(self) -> Any:
        return getattr(
            self.policy, "tokenizer_identity", getattr(self.policy, "tokenizer_info", None)
        )

    def _score(self, result: Any) -> Any:
        if hasattr(self.policy, "score_sampled_ids"):
            generation = SampledGeneration(
                text=result.response,
                prompt_token_ids=result.prompt_token_ids,
                token_ids=result.continuation_token_ids,
                token_offsets=result.continuation_token_offsets,
                policy=self._model_identity,
                tokenizer=self._tokenizer_identity,
            )
            return self.policy.score_sampled_ids(
                generation,
                require_grad=False,
                return_logits=False,
                return_logprobs=True,
            )
        from rlm_train.generation.generated_text import GeneratedText

        generated = GeneratedText(
            text=result.response,
            prompt_token_ids=result.prompt_token_ids,
            token_ids=result.continuation_token_ids,
            token_offsets=result.continuation_token_offsets,
            student=self._model_identity,
            tokenizer=self._tokenizer_identity,
        )
        return self.policy.score_tokens(
            generated,
            with_gradients=False,
            return_logits=False,
            return_logprobs=True,
        )


def _finite_logprob_tuple(values: Any) -> tuple[float, ...]:
    
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().tolist()
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("behavior token log probabilities must be non-empty and finite")
    return result


__all__ = ["TransformersAnswerSampler", "derive_matched_seed", "matched_seeds"]
