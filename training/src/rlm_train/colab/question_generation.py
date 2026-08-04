"""Generate one explicit, exactly traced helper-question edge for SDPO training."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from rlm.core.trajectory import CallItemSpan, InvocationKind

from rlm_train.colab.config import ColabRunConfig
from rlm_train.colab.generation import (
    TokenGenerationResult,
    TransformersResponseGenerator,
    derive_group_seed,
)
from rlm_train.trajectory import TrajectoryRecorder


class TracedQuestionResponseGenerator:
    """Sample a parent helper question, execute one child, and retain the real edge."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        configuration: ColabRunConfig,
    ) -> None:
        self.model_name = configuration.model.model_id
        self.rollout = configuration.sdpo_rollout
        parent_configuration = configuration.generation.model_copy(
            update={
                "system_prompt": self.rollout.question_system_prompt,
                "max_new_tokens": self.rollout.max_question_tokens,
            }
        )
        child_configuration = configuration.generation.model_copy(
            update={
                "system_prompt": self.rollout.child_system_prompt,
                "max_new_tokens": self.rollout.max_child_tokens,
            }
        )
        self.parent_generator = TransformersResponseGenerator(
            model,
            tokenizer,
            parent_configuration,
            model_context_length=configuration.model.max_context_length,
        )
        self.child_generator = TransformersResponseGenerator(
            model,
            tokenizer,
            child_configuration,
            model_context_length=configuration.model.max_context_length,
        )

    def generate_tokenized(
        self,
        prompt: str | dict[str, Any],
        *,
        seed: int,
        sample_index: int = 0,
    ) -> TokenGenerationResult:
        """Return exact parent tokens with a validated parent-to-child trajectory."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise TypeError("traced SDPO rollouts require a non-blank public text prompt")
        parent_messages = {
            "messages": [
                {"role": "system", "content": self.rollout.question_system_prompt},
                {"role": "user", "content": prompt},
            ]
        }
        parent = self.parent_generator.generate_tokenized(
            parent_messages,
            seed=seed,
            sample_index=sample_index,
        )
        if not parent.response.strip():
            raise RuntimeError("SDPO parent generated no visible helper question")

        child_prompt = f"Original problem:\n{prompt}\n\nHelper question:\n{parent.response}"
        child_messages = {
            "messages": [
                {"role": "system", "content": self.rollout.child_system_prompt},
                {"role": "user", "content": child_prompt},
            ]
        }
        child_seed = derive_group_seed(seed, "sdpo-child", sample_index)
        child = self.child_generator.generate_tokenized(
            child_messages,
            seed=child_seed,
            sample_index=sample_index,
        )

        identity_payload = f"{seed}\0{sample_index}\0{prompt}".encode()
        trajectory_id = hashlib.sha256(identity_payload).hexdigest()
        recorder = TrajectoryRecorder(
            trajectory_id,
            metadata={
                "schema": "single-question-edge-v1",
                "parent_seed": seed,
                "child_seed": child_seed,
            },
        )
        root_id = recorder.begin_node(
            kind=InvocationKind.ROOT,
            model=self.model_name,
            context=prompt,
            depth=0,
            policy_version=0,
        )
        child_id = recorder.begin_node(
            kind=InvocationKind.SUBCALL,
            model=self.model_name,
            context=child_prompt,
            parent_id=root_id,
            depth=1,
            call_order=0,
            policy_version=0,
        )
        recorder.complete_node(
            child_id,
            response=child.response,
            metadata=self._token_metadata(child),
        )
        recorder.complete_node(
            root_id,
            response=parent.response,
            call_item_spans=[
                CallItemSpan(
                    call_order=0,
                    batch_index=None,
                    start=0,
                    end=len(parent.response),
                    child_node_id=child_id,
                )
            ],
            metadata=self._token_metadata(parent),
        )
        return replace(parent, trajectory=recorder.snapshot())

    @staticmethod
    def _token_metadata(result: TokenGenerationResult) -> dict[str, Any]:
        """Return public exact-token metadata used for alignment and replay."""
        return {
            "prompt_token_ids": list(result.prompt_token_ids),
            "continuation_token_ids": list(result.continuation_token_ids),
            "continuation_token_offsets": [
                list(offset) for offset in result.continuation_token_offsets
            ],
            "prompt_length": result.prompt_length,
            "continuation_length": result.continuation_length,
            "termination_reason": result.termination_reason,
            "sampling": dict(result.sampling_metadata or {}),
        }


__all__ = ["TracedQuestionResponseGenerator"]
