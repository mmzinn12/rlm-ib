"""Exact-token Transformers generation for evaluation, rollouts, and RLM subcalls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


class GenerationConfig(BaseModel):
    """Exact prompt-formatting and sampling configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_template_version: str = "chat-v1"
    system_prompt: str = "Solve the problem carefully and give a final answer."
    use_chat_template: bool = True
    max_prompt_tokens: int = Field(default=512, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.8, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    do_sample: bool = True
    allow_prompt_truncation: bool = False

    @model_validator(mode="after")
    def validate_prompt_template(self) -> GenerationConfig:
        if not self.prompt_template_version.strip() or not self.system_prompt.strip():
            raise ValueError("prompt-template version and system prompt must not be blank")
        return self


@dataclass(frozen=True)
class TokenGenerationResult:
    """Retain exact sampled tensors instead of reconstructing them from text."""

    response: str
    prompt_token_ids: tuple[int, ...] = ()
    continuation_token_ids: tuple[int, ...] = ()
    continuation_token_offsets: tuple[tuple[int, int], ...] = ()
    prompt_length: int = 0
    continuation_length: int = 0
    sampling_metadata: dict[str, Any] | None = None


class PromptFormatter:
    """Own the one prompt template used by generation and policy/teacher forcing."""

    def __init__(self, tokenizer: Any, configuration: GenerationConfig) -> None:
        self.tokenizer = tokenizer
        self.configuration = configuration

    @property
    def fingerprint(self) -> str:
        """Hash the formatting policy and tokenizer chat template."""
        payload = {
            "version": self.configuration.prompt_template_version,
            "system_prompt": self.configuration.system_prompt,
            "use_chat_template": self.configuration.use_chat_template,
            "chat_template": getattr(self.tokenizer, "chat_template", None),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def messages(self, prompt: str | dict[str, Any]) -> list[dict[str, str]]:
        """Normalize a public string or message payload without verifier target data."""
        if isinstance(prompt, str):
            user_content = prompt
            messages = [
                {"role": "system", "content": self.configuration.system_prompt},
                {"role": "user", "content": user_content},
            ]
        elif isinstance(prompt, dict) and isinstance(prompt.get("messages"), list):
            messages = []
            for item in prompt["messages"]:
                if not isinstance(item, dict) or set(item) < {"role", "content"}:
                    raise ValueError("message prompts require role and content")
                messages.append({"role": str(item["role"]), "content": str(item["content"])})
        else:
            raise TypeError("prompt must be text or a mapping containing messages")
        if any(not message["content"].strip() for message in messages):
            raise ValueError("prompt messages must not be blank")
        return messages

    def encode_prompt(self, prompt: str | dict[str, Any]) -> tuple[int, ...]:
        """Tokenize through the shared versioned chat policy."""
        messages = self.messages(prompt)
        if self.configuration.use_chat_template:
            if not getattr(self.tokenizer, "chat_template", None):
                raise ValueError("use_chat_template requires a tokenizer chat template")
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            text = "\n".join(
                f"{message['role'].upper()}: {message['content']}" for message in messages
            )
            text = f"{text}\nASSISTANT:"
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("prompt formatter produced an unexpected batch")
            token_ids = token_ids[0]
        values = tuple(int(token_id) for token_id in token_ids)
        if not values:
            raise ValueError("prompt formatting produced no tokens")
        return values


class TransformersResponseGenerator:
    """Implement the generic evaluator generator contract with exact token boundaries."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        configuration: GenerationConfig,
        *,
        model_context_length: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.model_context_length = model_context_length
        self.formatter = PromptFormatter(tokenizer, configuration)
        if model_context_length <= 0:
            raise ValueError("model context length must be positive")

    def generate_tokenized(
        self,
        prompt: str | dict[str, Any],
        *,
        seed: int,
        sample_index: int = 0,
    ) -> TokenGenerationResult:
        """Sample one response while retaining model-returned token IDs exactly."""
        torch = _torch()
        if seed < 0 or sample_index < 0:
            raise ValueError("generation seed and sample index must be non-negative")
        prompt_ids = self.formatter.encode_prompt(prompt)
        prompt_truncated = False
        if len(prompt_ids) > self.configuration.max_prompt_tokens:
            if not self.configuration.allow_prompt_truncation:
                raise ValueError("formatted prompt exceeds configured max_prompt_tokens")
            prompt_ids = prompt_ids[-self.configuration.max_prompt_tokens :]
            prompt_truncated = True
        if len(prompt_ids) + self.configuration.max_new_tokens > self.model_context_length:
            raise ValueError("generation would exceed the configured model context length")
        device = model_device(self.model)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        generation_arguments: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.configuration.max_new_tokens,
            "do_sample": self.configuration.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.configuration.do_sample:
            generation_arguments.update(
                {
                    "temperature": self.configuration.temperature,
                    "top_p": self.configuration.top_p,
                }
            )
        was_training = bool(self.model.training)
        self.model.eval()
        device_type = device.type if hasattr(device, "type") else str(device).split(":", 1)[0]
        fork_devices = [device] if device_type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(seed)
                with torch.inference_mode():
                    generated = self.model.generate(**generation_arguments)
        finally:
            if was_training:
                self.model.train()
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError("model.generate must return one token sequence")
        all_ids = tuple(int(value) for value in generated[0].detach().cpu().tolist())
        if all_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("generated sequence does not preserve the exact prompt prefix")
        continuation_ids = all_ids[len(prompt_ids) :]
        if not continuation_ids:
            raise RuntimeError("model generated an empty continuation")
        eos_token_id = self.tokenizer.eos_token_id
        reached_eos = eos_token_id is not None and continuation_ids[-1] == eos_token_id
        reached_limit = len(continuation_ids) >= self.configuration.max_new_tokens
        response = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        continuation_offsets = decode_token_offsets(
            self.tokenizer,
            continuation_ids,
            expected_text=response,
        )
        sampling_metadata = {
            "seed": seed,
            "sample_index": sample_index,
            "do_sample": self.configuration.do_sample,
            "temperature": self.configuration.temperature,
            "top_p": self.configuration.top_p,
            "max_new_tokens": self.configuration.max_new_tokens,
            "prompt_template_fingerprint": self.formatter.fingerprint,
            "termination_reason": (
                "eos" if reached_eos else "length" if reached_limit else "stopped"
            ),
            "prompt_truncated": prompt_truncated,
        }
        return TokenGenerationResult(
            response=response,
            prompt_token_ids=prompt_ids,
            continuation_token_ids=continuation_ids,
            continuation_token_offsets=continuation_offsets,
            prompt_length=len(prompt_ids),
            continuation_length=len(continuation_ids),
            sampling_metadata=sampling_metadata,
        )


class TransformersCompletionAdapter(BaseLM):
    """Expose exact-token local generation to the RLM client boundary."""

    def __init__(
        self,
        generator: TransformersResponseGenerator,
        *,
        model_name: str,
        base_seed: int,
    ) -> None:
        super().__init__(model_name=model_name)
        if base_seed < 0:
            raise ValueError("base seed must be non-negative")
        self.generator = generator
        self.base_seed = base_seed
        self._call_count = 0
        self._generation_lock = threading.Lock()
        self._last_usage = ModelUsageSummary(0, 0, 0, 0.0)
        self._usage = ModelUsageSummary(0, 0, 0, 0.0)
        self._generation_local = threading.local()

    @property
    def policy_owner(self) -> str:
        return f"student:{self.model_name}"

    def completion(self, prompt: str | dict[str, Any]) -> str:
        """Generate and record one root or explicitly scoped subcall."""
        result = self.generate_completion(prompt)
        return result.response

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        """Run the synchronous CUDA generation without duplicating logic."""
        return await asyncio.to_thread(self.completion, prompt)

    async def acompletion_with_generation(
        self, prompt: str | dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Keep async generation metadata paired with its originating worker call."""
        result = await asyncio.to_thread(self.generate_completion, prompt)
        return result.response, {
            "prompt_token_ids": result.prompt_token_ids,
            "token_ids": result.continuation_token_ids,
            "token_offsets": result.continuation_token_offsets,
            "prompt_token_count": result.prompt_length,
            "policy_owner": self.policy_owner,
        }

    def generate_completion(self, prompt: str | dict[str, Any]) -> TokenGenerationResult:
        """Generate one serialized model completion and retain exact metadata."""
        with self._generation_lock:
            seed = derive_group_seed(self.base_seed, str(prompt), self._call_count)
            self._call_count += 1
            result = self.generator.generate_tokenized(prompt, seed=seed, sample_index=0)
            self._generation_local.value = result
            input_tokens = result.prompt_length
            output_tokens = result.continuation_length
            self._last_usage = ModelUsageSummary(1, input_tokens, output_tokens, 0.0)
            self._usage.total_calls += 1
            self._usage.total_input_tokens += input_tokens
            self._usage.total_output_tokens += output_tokens
        return result

    def get_usage_summary(self) -> UsageSummary:
        """Return cumulative local token counts with zero API cost."""
        return UsageSummary({self.model_name: self._usage})

    def get_last_usage(self) -> ModelUsageSummary:
        """Return token counts for the most recent local generation."""
        return self._last_usage

    def get_last_generation(self) -> dict[str, Any] | None:
        """Expose exact sampled IDs to canonical RLM instrumentation."""
        result = getattr(self._generation_local, "value", None)
        if result is None:
            return None
        return {
            "prompt_token_ids": result.prompt_token_ids,
            "token_ids": result.continuation_token_ids,
            "token_offsets": result.continuation_token_offsets,
            "prompt_token_count": result.prompt_length,
            "policy_owner": self.policy_owner,
        }


def continuation_logprobs(
    model: Any,
    *,
    prompt_token_ids: tuple[int, ...],
    continuation_token_ids: tuple[int, ...],
    require_grad: bool,
) -> Any:
    """Score exact continuation IDs under a causal model with prompt tokens masked."""
    torch = _torch()
    if not prompt_token_ids or not continuation_token_ids:
        raise ValueError("log-probability scoring requires prompt and continuation tokens")
    continuation_logits = score_continuation_logits(
        model,
        prompt_token_ids=prompt_token_ids,
        continuation_token_ids=continuation_token_ids,
        require_grad=require_grad,
    )
    targets = torch.tensor(
        continuation_token_ids,
        dtype=torch.long,
        device=continuation_logits.device,
    )
    logprobs = torch.log_softmax(continuation_logits.float(), dim=-1)
    selected = logprobs.gather(dim=-1, index=targets[:, None]).squeeze(-1)
    return selected if require_grad else selected.detach()


def score_continuation_logits(
    model: Any,
    *,
    prompt_token_ids: tuple[int, ...],
    continuation_token_ids: tuple[int, ...],
    require_grad: bool,
) -> Any:
    """Return full-vocabulary logits aligned to exact sampled continuation IDs."""
    torch = _torch()
    if not prompt_token_ids or not continuation_token_ids:
        raise ValueError("logit scoring requires prompt and continuation tokens")
    device = model_device(model)
    complete = (*prompt_token_ids, *continuation_token_ids)
    input_ids = torch.tensor([complete], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError("causal model logits must align with input IDs")
        start = len(prompt_token_ids) - 1
        stop = start + len(continuation_token_ids)
        continuation_logits = logits[0, start:stop, :]
    return continuation_logits if require_grad else continuation_logits.detach()


def derive_group_seed(base_seed: int, prompt_identity: str, sample_index: int) -> int:
    """Derive stable independent seeds for grouped and resumable rollouts."""
    if base_seed < 0 or sample_index < 0:
        raise ValueError("seed inputs must be non-negative")
    payload = f"{base_seed}\0{prompt_identity}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def decode_token_offsets(
    tokenizer: Any,
    token_ids: tuple[int, ...],
    *,
    expected_text: str,
) -> tuple[tuple[int, int], ...]:
    """Map exact sampled IDs to visible character spans without re-tokenizing text."""
    offsets: list[tuple[int, int]] = []
    previous = ""
    for position in range(len(token_ids)):
        prefix_ids = token_ids[: position + 1]
        try:
            current = tokenizer.decode(
                prefix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            current = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        if not current.startswith(previous):
            raise ValueError("tokenizer prefix decoding cannot produce stable character offsets")
        offsets.append((len(previous), len(current)))
        previous = current
    if previous != expected_text:
        raise ValueError("exact token offsets do not reconstruct the generated response")
    return tuple(offsets)


def model_device(model: Any) -> Any:
    """Resolve a module device and fail when no parameters or buffers exist."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration as exc:
            raise ValueError("model has no parameters or buffers") from exc
    except AttributeError as exc:
        raise TypeError("model must be a PyTorch module") from exc


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Transformers generation") from exc


__all__ = [
    "PromptFormatter",
    "TokenGenerationResult",
    "TransformersCompletionAdapter",
    "TransformersResponseGenerator",
    "continuation_logprobs",
    "decode_token_offsets",
    "derive_group_seed",
    "model_device",
    "score_continuation_logits",
]
