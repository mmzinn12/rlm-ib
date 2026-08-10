"""Construct trainable policies from a StudentSpec.

The policy is the student model that generates rollouts and is scored during training. Building
one loads a Hugging Face causal LM plus tokenizer, so these entry points require the
``transformers``/``torch`` stack and, in practice, a GPU/Colab runtime. ``build_policy`` is the
generic entry point that dispatches on ``StudentSpec.adapter``; ``build_transformers_policy`` is
the concrete Transformers implementation.
"""

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
    """Load a Hugging Face causal LM and tokenizer into a trainable TransformersPolicy.

    Args:
        student: Student model specification (model/tokenizer ids, revisions, trust flag).
        runtime: Runtime settings; ``precision`` selects the model dtype and ``seed`` the base seed.
        checkpoint_id: Identifier recorded on the policy identity for provenance.
        generation: Optional generation/prompt-formatting config; defaults to a fresh one.

    Returns:
        A ``TransformersPolicy`` wrapping the loaded model, placed on CUDA when available.

    Raises:
        RuntimeError: If ``transformers`` or ``torch`` are not importable.
    """
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


def build_policy(
    student: StudentSpec, *, runtime: RuntimeSpec, checkpoint_id: str = "latest"
) -> TransformersPolicy:
    """Build a trainable policy by dispatching on the student adapter.

    Args:
        student: Student model specification; ``adapter`` selects the concrete builder.
        runtime: Runtime settings forwarded to the concrete builder.
        checkpoint_id: Identifier recorded on the policy identity for provenance.

    Returns:
        The constructed trainable policy for the requested adapter.

    Raises:
        NotImplementedError: If ``student.adapter`` has no wired builder.
    """
    if student.adapter == "transformers":
        return build_transformers_policy(student, runtime=runtime, checkpoint_id=checkpoint_id)
    raise NotImplementedError(f"policy adapter {student.adapter!r} is not wired yet")


__all__ = ["build_policy", "build_transformers_policy"]
