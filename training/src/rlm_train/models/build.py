"""Load a Hugging Face causal LM into a TransformersPolicy. Requires a GPU/Colab runtime."""

from __future__ import annotations

from typing import Any

from rlm_train.models.identity import PolicyIdentity, TokenizerIdentity
from rlm_train.models.transformers import TransformersPolicy, TransformersResponseGenerator
from rlm_train.models.transformers_runtime import GenerationConfig
from rlm_train.spec.models import StudentSpec
from rlm_train.spec.run import RuntimeSpec

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def build_transformers_policy(
    student: StudentSpec,
    *,
    runtime: RuntimeSpec,
    checkpoint_id: str = "latest",
    generation: GenerationConfig | None = None,
) -> TransformersPolicy:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required to load a student policy") from exc

    dtype = getattr(torch, _DTYPES.get(runtime.precision, "float32"))
    tokenizer_id = student.tokenizer_id or student.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=student.tokenizer_revision,
        trust_remote_code=student.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        student.model_id,
        revision=student.revision,
        trust_remote_code=student.trust_remote_code,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    context_length = int(getattr(model.config, "max_position_embeddings", 0) or 4096)
    generator = TransformersResponseGenerator(
        model,
        tokenizer,
        generation or GenerationConfig(),
        model_context_length=context_length,
    )
    identity = PolicyIdentity(
        component_id=student.model_id,
        revision=student.revision or "default",
        policy_owner=student.resolved_policy_owner,
        checkpoint_id=checkpoint_id,
    )
    tokenizer_identity = TokenizerIdentity(
        component_id=tokenizer_id,
        revision=student.tokenizer_revision or "default",
        vocabulary_size=int(getattr(tokenizer, "vocab_size", 0)) or None,
    )
    return TransformersPolicy(
        generator,
        identity=identity,
        tokenizer_identity=tokenizer_identity,
        base_seed=runtime.seed,
    )


__all__ = ["build_transformers_policy"]
