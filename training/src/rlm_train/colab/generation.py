"""Exact-token Transformers generation for evaluation, rollouts, and RLM subcalls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.trajectory import InvocationKind
from rlm.core.types import ModelUsageSummary, UsageSummary

from rlm_train.benchmarks import GenerationResult, ModelProvenance
from rlm_train.colab.config import GenerationConfig
from rlm_train.trajectory import TrajectoryRecorder


@dataclass(frozen=True)
class TokenGenerationResult(GenerationResult):
    """Retain exact sampled tensors instead of reconstructing them from text."""

    prompt_token_ids: tuple[int, ...] = ()
    continuation_token_ids: tuple[int, ...] = ()
    continuation_token_offsets: tuple[tuple[int, int], ...] = ()
    attention_mask: tuple[int, ...] = ()
    prompt_length: int = 0
    continuation_length: int = 0
    termination_reason: str = "unknown"
    sampling_metadata: dict[str, Any] | None = None
    prompt_truncated: bool = False


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

    async def generate(
        self,
        *,
        prompt: str,
        public_problem: dict[str, Any],
        seed: int,
        sample_index: int,
        model: ModelProvenance,
    ) -> TokenGenerationResult:
        """Generate one evaluator response without accepting verifier target data."""
        if "target" in public_problem or "target_data" in public_problem:
            raise ValueError("verifier target crossed into the response generator")
        del model
        return self.generate_tokenized(prompt, seed=seed, sample_index=sample_index)

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
        generator_device = device.type if hasattr(device, "type") else str(device)
        random_generator = torch.Generator(device=generator_device)
        random_generator.manual_seed(seed)
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
                    "generator": random_generator,
                }
            )
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.inference_mode():
            generated = self.model.generate(**generation_arguments)
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
        termination_reason = "eos" if reached_eos else "length" if reached_limit else "stopped"
        response = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        continuation_offsets = decode_token_offsets(
            self.tokenizer,
            continuation_ids,
            expected_text=response,
        )
        response_tokens = tuple(self.tokenizer.convert_ids_to_tokens(list(continuation_ids)))
        sampling_metadata = {
            "seed": seed,
            "sample_index": sample_index,
            "do_sample": self.configuration.do_sample,
            "temperature": self.configuration.temperature,
            "top_p": self.configuration.top_p,
            "max_new_tokens": self.configuration.max_new_tokens,
            "prompt_template_fingerprint": self.formatter.fingerprint,
        }
        return TokenGenerationResult(
            response=response,
            response_tokens=response_tokens,
            token_count=len(continuation_ids),
            truncated=reached_limit and not reached_eos,
            prompt_token_ids=prompt_ids,
            continuation_token_ids=continuation_ids,
            continuation_token_offsets=continuation_offsets,
            attention_mask=(1,) * len(all_ids),
            prompt_length=len(prompt_ids),
            continuation_length=len(continuation_ids),
            termination_reason=termination_reason,
            sampling_metadata=sampling_metadata,
            prompt_truncated=prompt_truncated,
        )


class TransformersCompletionAdapter(BaseLM):
    """Expose the local generator to RLM LMHandler calls with trajectory recording."""

    def __init__(
        self,
        generator: TransformersResponseGenerator,
        *,
        model_name: str,
        base_seed: int,
        recorder: TrajectoryRecorder | None = None,
        policy_version: int = 0,
    ) -> None:
        super().__init__(model_name=model_name)
        if base_seed < 0 or policy_version < 0:
            raise ValueError("base seed and policy version must be non-negative")
        self.generator = generator
        self.base_seed = base_seed
        self.recorder = recorder
        self.policy_version = policy_version
        self._call_count = 0
        self._last_usage = ModelUsageSummary(0, 0, 0, 0.0)
        self._usage = ModelUsageSummary(0, 0, 0, 0.0)
        self._span_local = threading.local()

    @contextmanager
    def span_context(
        self,
        *,
        parent_node_id: str,
        depth: int,
        call_order: int,
        batch_index: int | None = None,
    ):
        """Mark the next completion as a specific recursive child span."""
        previous = getattr(self._span_local, "value", None)
        self._span_local.value = (parent_node_id, depth, call_order, batch_index)
        try:
            yield
        finally:
            self._span_local.value = previous

    def completion(self, prompt: str | dict[str, Any]) -> str:
        """Generate and record one root or explicitly scoped subcall."""
        result = self.generate_for_span(prompt)
        return result.response

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        """Run the synchronous CUDA generation without duplicating logic."""
        return await asyncio.to_thread(self.completion, prompt)

    def generate_for_span(
        self,
        prompt: str | dict[str, Any],
        *,
        parent_node_id: str | None = None,
        depth: int = 0,
        call_order: int | None = None,
        batch_index: int | None = None,
    ) -> TokenGenerationResult:
        """Generate a response and persist exact root/subcall token metadata."""
        implicit = getattr(self._span_local, "value", None)
        if implicit is not None and parent_node_id is None:
            parent_node_id, depth, call_order, batch_index = implicit
        seed = derive_group_seed(self.base_seed, str(prompt), self._call_count)
        self._call_count += 1
        result = self.generator.generate_tokenized(prompt, seed=seed, sample_index=0)
        input_tokens = result.prompt_length
        output_tokens = result.continuation_length
        self._last_usage = ModelUsageSummary(1, input_tokens, output_tokens, 0.0)
        self._usage.total_calls += 1
        self._usage.total_input_tokens += input_tokens
        self._usage.total_output_tokens += output_tokens
        if self.recorder is not None:
            kind = InvocationKind.ROOT if parent_node_id is None else InvocationKind.SUBCALL
            if kind is InvocationKind.SUBCALL and call_order is None:
                raise ValueError("recorded subcalls require call_order")
            node_id = self.recorder.begin_node(
                kind=kind,
                model=self.model_name,
                context=prompt,
                parent_id=parent_node_id,
                depth=depth,
                call_order=call_order,
                batch_index=batch_index,
                policy_version=self.policy_version,
            )
            self.recorder.complete_node(
                node_id,
                response=result.response,
                metadata={
                    "prompt_token_ids": list(result.prompt_token_ids),
                    "continuation_token_ids": list(result.continuation_token_ids),
                    "continuation_token_offsets": [
                        list(offset) for offset in result.continuation_token_offsets
                    ],
                    "prompt_length": result.prompt_length,
                    "continuation_length": result.continuation_length,
                    "termination_reason": result.termination_reason,
                    "sampling": result.sampling_metadata,
                },
            )
        return result

    def get_usage_summary(self) -> UsageSummary:
        """Return cumulative local token counts with zero API cost."""
        return UsageSummary({self.model_name: self._usage})

    def get_last_usage(self) -> ModelUsageSummary:
        """Return token counts for the most recent local generation."""
        return self._last_usage


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
