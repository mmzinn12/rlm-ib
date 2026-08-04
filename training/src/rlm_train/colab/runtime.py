"""Preflight and load a reproducible LoRA model for one CUDA device."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rlm_train.colab.config import (
    ColabRunConfig,
    Precision,
    Quantization,
)
from rlm_train.experiment.config import TrainingAlgorithm

_DEPENDENCIES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "pydantic",
    "rlm-train",
    "rlms",
)


class CUDAProbe(Protocol):
    """Narrow CUDA surface used by preflight and unit tests."""

    def is_available(self) -> bool: ...

    def is_bf16_supported(self) -> bool: ...

    def get_device_name(self, device: int = 0) -> str: ...


@dataclass(frozen=True)
class RuntimePreflight:
    """Record checks completed before any model download begins."""

    cuda_device_name: str
    precision: str
    output_directory: str
    dependency_versions: dict[str, str]
    python_version: str


@dataclass(frozen=True)
class ModelBundle:
    """Hold the exact tokenizer, LoRA policy, and startup provenance."""

    model: Any
    tokenizer: Any
    tokenizer_fingerprint: str
    model_revision: str
    tokenizer_revision: str
    provenance: dict[str, Any]


def validate_colab_runtime(
    configuration: ColabRunConfig,
    *,
    cuda: CUDAProbe | None = None,
    environment: dict[str, str] | None = None,
    base_directory: str | Path | None = None,
    installed_versions: dict[str, str] | None = None,
) -> RuntimePreflight:
    """Fail before model loading when the single-GPU runtime is incompatible."""
    torch = _import_torch()
    probe = cuda or torch.cuda
    if not probe.is_available():
        raise RuntimeError("the Colab training path requires an available CUDA device")
    if configuration.model.precision is Precision.BF16 and not probe.is_bf16_supported():
        raise RuntimeError("requested bf16 precision is unsupported by this CUDA device")
    if configuration.model.precision is Precision.FP16:
        device_name = probe.get_device_name(0)
        if not device_name.strip():
            raise RuntimeError("CUDA device name could not be resolved")
    else:
        device_name = probe.get_device_name(0)
    if not device_name.strip():
        raise RuntimeError("CUDA device name could not be resolved")
    experiment = configuration.resolved_experiment
    if (
        experiment.training.algorithm is TrainingAlgorithm.SDPO
        and configuration.judge.provider == "openai"
    ):
        values = os.environ if environment is None else environment
        if not values.get(configuration.judge.api_key_environment, "").strip():
            raise RuntimeError(
                f"required judge secret {configuration.judge.api_key_environment!r} is absent"
            )
    if configuration.execution_backend != "transformers":
        raise ValueError("the Colab entry point supports only the local Transformers backend")
    versions = dependency_versions() if installed_versions is None else dict(installed_versions)
    required = {"torch", "transformers", "peft", "accelerate", "safetensors"}
    if configuration.model.quantization is not Quantization.NONE:
        required.add("bitsandbytes")
    if (
        experiment.training.algorithm is TrainingAlgorithm.SDPO
        and configuration.judge.provider == "openai"
    ):
        required.add("openai")
    missing = sorted(
        name for name in required if versions.get(name, "not-installed") == "not-installed"
    )
    if missing:
        raise RuntimeError(f"required Colab dependencies are not installed: {missing!r}")
    if configuration.output.google_drive_root is not None:
        drive_root = Path(configuration.output.google_drive_root).expanduser()
        if not drive_root.is_dir():
            raise RuntimeError(f"configured Google Drive root is not mounted: {drive_root}")
    output = configuration.output.resolve_directory(base_directory=base_directory)
    return RuntimePreflight(
        cuda_device_name=device_name,
        precision=configuration.model.precision.value,
        output_directory=str(output),
        dependency_versions=versions,
        python_version=platform.python_version(),
    )


def dependency_versions() -> dict[str, str]:
    """Return installed versions without importing optional training packages."""
    versions: dict[str, str] = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash vocabulary, special tokens, chat template, class, and source identity."""
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("tokenizer must expose a non-empty vocabulary")
    payload = {
        "class": type(tokenizer).__qualname__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocabulary": sorted((str(token), int(index)) for token, index in vocabulary.items()),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "commit_hash": getattr(tokenizer, "_commit_hash", None),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_policy_bundle(
    configuration: ColabRunConfig,
    preflight: RuntimePreflight,
) -> ModelBundle:
    """Load a causal LM, attach LoRA, and return exact resolved identities."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "install the 'colab' optional dependency before loading a Transformers policy"
        ) from exc
    torch = _import_torch()
    model_configuration = configuration.model
    random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)
    torch.cuda.manual_seed_all(configuration.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_configuration.resolved_tokenizer_id,
        revision=model_configuration.resolved_tokenizer_revision,
        trust_remote_code=model_configuration.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = precision_dtype(model_configuration.precision)
    load_arguments: dict[str, Any] = {
        "revision": model_configuration.model_revision,
        "trust_remote_code": model_configuration.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": {"": 0},
    }
    if model_configuration.quantization is not Quantization.NONE:
        load_arguments["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=model_configuration.quantization is Quantization.INT4,
            load_in_8bit=model_configuration.quantization is Quantization.INT8,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_configuration.model_id,
        **load_arguments,
    )
    if model_configuration.quantization is not Quantization.NONE:
        model = prepare_model_for_kbit_training(model)
    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_configuration.lora_rank,
        lora_alpha=model_configuration.lora_alpha,
        lora_dropout=model_configuration.lora_dropout,
        target_modules=list(model_configuration.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, adapter)
    model.train()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attachment produced no trainable student parameters")
    resolved_model_revision = str(
        getattr(model.config, "_commit_hash", None)
        or getattr(model, "_commit_hash", None)
        or model_configuration.model_revision
    )
    resolved_tokenizer_revision = str(
        getattr(tokenizer, "_commit_hash", None) or model_configuration.resolved_tokenizer_revision
    )
    fingerprint = tokenizer_fingerprint(tokenizer)
    provenance = {
        "model_id": model_configuration.model_id,
        "requested_model_revision": model_configuration.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "tokenizer_id": model_configuration.resolved_tokenizer_id,
        "requested_tokenizer_revision": model_configuration.resolved_tokenizer_revision,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "tokenizer_fingerprint": fingerprint,
        "precision": model_configuration.precision.value,
        "quantization": model_configuration.quantization.value,
        "lora": {
            "rank": model_configuration.lora_rank,
            "alpha": model_configuration.lora_alpha,
            "dropout": model_configuration.lora_dropout,
            "target_modules": list(model_configuration.lora_target_modules),
            "trainable_parameters": trainable,
        },
        "cuda_device_name": preflight.cuda_device_name,
        "dependency_versions": preflight.dependency_versions,
        "resolved_output_directory": preflight.output_directory,
    }
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        tokenizer_fingerprint=fingerprint,
        model_revision=resolved_model_revision,
        tokenizer_revision=resolved_tokenizer_revision,
        provenance=provenance,
    )


def precision_dtype(precision: Precision) -> Any:
    """Map the validated precision enum to a PyTorch dtype."""
    torch = _import_torch()
    return {
        Precision.FP32: torch.float32,
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
    }[precision]


def _import_torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Colab training path") from exc


__all__ = [
    "CUDAProbe",
    "ModelBundle",
    "RuntimePreflight",
    "dependency_versions",
    "load_policy_bundle",
    "precision_dtype",
    "tokenizer_fingerprint",
    "validate_colab_runtime",
]
