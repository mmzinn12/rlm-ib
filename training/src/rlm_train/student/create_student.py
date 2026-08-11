"""Load the shared trainable student model and tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlm_train.generation.generate import TransformersGenerator
from rlm_train.generation.settings import GenerationSettings
from rlm_train.settings.run import RuntimeSettings
from rlm_train.settings.student import StudentSettings
from rlm_train.student.model_info import StudentModelInfo, TokenizerInfo
from rlm_train.student.transformers_student import TransformersStudent

DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def create_transformers_student(
    settings: StudentSettings,
    *,
    runtime: RuntimeSettings,
    checkpoint_id: str = "latest",
    generation: GenerationSettings | None = None,
    checkpoint_path: str | Path | None = None,
) -> TransformersStudent:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required to load a student") from exc

    dtype = getattr(torch, DTYPES.get(runtime.precision, "float32"))
    model_source = str(checkpoint_path) if checkpoint_path is not None else settings.model_id
    tokenizer_source = (
        str(checkpoint_path)
        if checkpoint_path is not None
        else settings.tokenizer_id or settings.model_id
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if checkpoint_path is not None else settings.tokenizer_revision,
        trust_remote_code=settings.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=None if checkpoint_path is not None else settings.revision,
        trust_remote_code=settings.trust_remote_code,
        dtype=dtype,
    )
    if settings.trainable:
        model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model = model.to("cuda")
    context_length = settings.generation.model_context_length or int(
        getattr(model.config, "max_position_embeddings", 0) or 4096
    )
    generation_settings = generation or GenerationSettings(
        max_prompt_tokens=settings.generation.max_prompt_tokens,
        max_new_tokens=settings.generation.max_new_tokens,
        temperature=settings.generation.temperature,
        top_p=settings.generation.top_p,
        do_sample=settings.generation.do_sample,
        use_chat_template=settings.generation.use_chat_template,
        allow_prompt_truncation=settings.generation.allow_prompt_truncation,
    )
    generator = TransformersGenerator(
        model,
        tokenizer,
        generation_settings,
        model_context_length=context_length,
    )
    return TransformersStudent(
        generator,
        model_info=StudentModelInfo(
            component_id=settings.model_id,
            revision=settings.revision or "default",
            student_id=settings.resolved_policy_owner,
            checkpoint_id=(
                Path(checkpoint_path).name if checkpoint_path is not None else checkpoint_id
            ),
        ),
        tokenizer_info=TokenizerInfo(
            component_id=tokenizer_source,
            revision=settings.tokenizer_revision or "default",
            vocabulary_size=int(getattr(tokenizer, "vocab_size", 0)) or None,
        ),
        base_seed=runtime.seed,
    )


def create_student(
    settings: StudentSettings,
    *,
    runtime: RuntimeSettings,
    checkpoint_id: str = "latest",
    checkpoint_path: str | Path | None = None,
) -> TransformersStudent:
    if settings.adapter != "transformers":
        raise NotImplementedError(f"student adapter {settings.adapter!r} is not wired yet")
    return create_transformers_student(
        settings,
        runtime=runtime,
        checkpoint_id=checkpoint_id,
        checkpoint_path=checkpoint_path,
    )


__all__ = ["create_student", "create_transformers_student"]
