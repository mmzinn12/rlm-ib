"""Exact-token fixed-teacher scoring with explicit single-GPU residency."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm_train.colab.config import TeacherResidency
from rlm_train.colab.generation import PromptFormatter, model_device
from rlm_train.colab.runtime import tokenizer_fingerprint
from rlm_train.judge import QuestionTeacherFeedback
from rlm_train.sdpo import (
    TopKTeacherTarget,
    TorchFixedTeacherController,
    build_question_feedback_context,
    extract_topk_teacher_target,
)


@dataclass(frozen=True)
class TeacherTargetProvenance:
    """Identify every alignment-sensitive input to one cached target."""

    cache_key: str
    teacher_fingerprint: str
    tokenizer_fingerprint: str
    prompt_template_fingerprint: str
    feedback_fingerprint: str
    continuation_fingerprint: str
    residency: str


class FileTeacherTargetCache:
    """Persist immutable compact targets as content-addressed JSON files."""

    def __init__(self, directory: str | Path) -> None:
        if not str(directory).strip():
            raise ValueError("teacher target cache directory must not be blank")
        self.directory = Path(directory)

    def get(self, key: str) -> TopKTeacherTarget | None:
        """Read one validated target without exposing unhashed cache inputs."""
        path = self._path(key)
        if not path.exists():
            return None
        return TopKTeacherTarget.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, target: TopKTeacherTarget) -> None:
        """Atomically write one target and reject conflicting content."""
        path = self._path(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = f"{target.model_dump_json()}\n"
        if path.exists():
            existing = TopKTeacherTarget.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != target:
                raise ValueError("teacher cache key collision contains a different target")
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def manifest(self) -> dict[str, Any]:
        """Return only content keys and counts, never prompts or feedback."""
        keys = sorted(path.stem for path in self.directory.glob("*.json"))
        return {"format": "topk-teacher-target-v1", "count": len(keys), "keys": keys}

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("teacher target cache key must be a SHA-256 hex digest")
        return self.directory / f"{key}.json"


class TransformersQuestionTeacherProvider:
    """Teacher-force exact sampled IDs under feedback-conditioned fixed-teacher prompts."""

    def __init__(
        self,
        controller: TorchFixedTeacherController,
        tokenizer: Any,
        formatter: PromptFormatter,
        *,
        student_tokenizer_fingerprint: str,
        residency: TeacherResidency,
        top_k: int,
        cache: FileTeacherTargetCache | None = None,
        student_model: Any | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("teacher top_k must be positive")
        self.controller = controller
        self.teacher = controller.teacher
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.residency = TeacherResidency(residency)
        self.top_k = top_k
        self.cache = cache
        self.student_model = student_model
        self.tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)
        if self.tokenizer_fingerprint != student_tokenizer_fingerprint:
            raise ValueError("fixed teacher and student tokenizer fingerprints differ")
        if self.residency is TeacherResidency.SEQUENTIAL and student_model is None:
            raise ValueError("sequential teacher residency requires the student model")
        if self.residency in (TeacherResidency.CPU_OFFLOAD, TeacherResidency.SEQUENTIAL):
            self.teacher.to("cpu")
        self.controller.validate_unchanged()

    @property
    def teacher_version(self) -> int:
        """A fixed teacher is always version zero."""
        return 0

    @property
    def identity(self) -> dict[str, Any]:
        """Return model and placement provenance without duplicating weights."""
        payload = self.controller.identity().to_dict()
        payload.update(
            {
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "prompt_template_fingerprint": self.formatter.fingerprint,
                "residency": self.residency.value,
            }
        )
        return payload

    def render_teacher_prompt(
        self,
        *,
        original_question: str,
        feedback: QuestionTeacherFeedback,
    ) -> str:
        """Render only the public question and the selected restricted projection."""
        if not original_question.strip():
            raise ValueError("original teacher question must not be blank")
        restricted = build_question_feedback_context(feedback)
        payload = json.dumps(
            restricted,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return (
            f"Original question:\n{original_question}\n\n"
            "Edge-local teacher feedback (the only assessment context provided):\n"
            f"{payload}\n\n"
            "Score the existing student continuation; do not generate a replacement."
        )

    def score_target(
        self,
        *,
        original_question: str,
        continuation_token_ids: tuple[int, ...],
        feedback: QuestionTeacherFeedback,
    ) -> tuple[TopKTeacherTarget, TeacherTargetProvenance]:
        """Return an aligned compact target, reusing only an exact immutable cache key."""
        if not continuation_token_ids or any(token_id < 0 for token_id in continuation_token_ids):
            raise ValueError("teacher forcing requires non-empty valid continuation token IDs")
        prompt = self.render_teacher_prompt(
            original_question=original_question,
            feedback=feedback,
        )
        prompt_token_ids = self.formatter.encode_prompt(prompt)
        vocabulary_size = int(getattr(self.teacher.config, "vocab_size", 0))
        if vocabulary_size <= max(continuation_token_ids):
            raise ValueError("sampled continuation contains a token outside teacher vocabulary")
        provenance = self._target_provenance(
            prompt_token_ids=prompt_token_ids,
            continuation_token_ids=continuation_token_ids,
            feedback=feedback,
        )
        if self.cache is not None:
            cached = self.cache.get(provenance.cache_key)
            if cached is not None:
                return cached, provenance
        logits = self._score_ids(prompt_token_ids, continuation_token_ids)
        if logits.shape[0] != len(continuation_token_ids):
            raise RuntimeError("teacher logits do not align with sampled continuation IDs")
        target = extract_topk_teacher_target(
            logits,
            top_k=self.top_k,
            teacher_version=0,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
        )
        if self.cache is not None:
            self.cache.put(provenance.cache_key, target)
        self.controller.validate_unchanged()
        return target, provenance

    async def score_existing_continuation(
        self,
        *,
        student_context: Any,
        student_continuation: str,
        feedback: QuestionTeacherFeedback,
    ) -> Any:
        """Compatibility method for the legacy text-bound scorer protocol.

        This path validates round-trip tokenization. The Colab trainer calls
        :meth:`score_target` with sampled IDs and never uses this compatibility method.
        """
        if not isinstance(student_context, str):
            raise TypeError("text compatibility scoring requires a string student context")
        continuation_ids = tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(
                student_continuation,
                add_special_tokens=False,
            )
        )
        decoded = self.tokenizer.decode(continuation_ids, skip_special_tokens=False)
        if decoded != student_continuation:
            raise ValueError("text continuation does not round-trip to exact teacher token IDs")
        prompt = self.render_teacher_prompt(
            original_question=student_context,
            feedback=feedback,
        )
        return self._score_ids(self.formatter.encode_prompt(prompt), continuation_ids)

    def _score_ids(
        self,
        prompt_token_ids: tuple[int, ...],
        continuation_token_ids: tuple[int, ...],
    ) -> Any:
        torch = _torch()
        original_student_device = None
        if self.residency is TeacherResidency.SEQUENTIAL:
            assert self.student_model is not None
            original_student_device = model_device(self.student_model)
            self.student_model.to("cpu")
            torch.cuda.empty_cache()
            self.teacher.to(original_student_device)
        teacher_device = model_device(self.teacher)
        complete = (*prompt_token_ids, *continuation_token_ids)
        input_ids = torch.tensor([complete], dtype=torch.long, device=teacher_device)
        attention_mask = torch.ones_like(input_ids)
        self.teacher.eval()
        try:
            with torch.inference_mode():
                output = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                start = len(prompt_token_ids) - 1
                stop = start + len(continuation_token_ids)
                logits = output.logits[0, start:stop, :].detach().float().cpu()
        finally:
            if self.residency is TeacherResidency.SEQUENTIAL:
                assert original_student_device is not None
                self.teacher.to("cpu")
                torch.cuda.empty_cache()
                self.student_model.to(original_student_device)
        if not torch.isfinite(logits).all().item():
            raise ValueError("teacher produced non-finite continuation logits")
        return logits

    def _target_provenance(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        continuation_token_ids: tuple[int, ...],
        feedback: QuestionTeacherFeedback,
    ) -> TeacherTargetProvenance:
        feedback_json = feedback.model_dump_json()
        feedback_fingerprint = hashlib.sha256(feedback_json.encode()).hexdigest()
        continuation_json = json.dumps(continuation_token_ids, separators=(",", ":"))
        continuation_fingerprint = hashlib.sha256(continuation_json.encode()).hexdigest()
        payload = {
            "teacher": self.controller.initial_fingerprint,
            "teacher_version": 0,
            "tokenizer": self.tokenizer_fingerprint,
            "prompt_template": self.formatter.fingerprint,
            "prompt_token_ids": prompt_token_ids,
            "feedback": feedback.model_dump(mode="json"),
            "continuation_token_ids": continuation_token_ids,
            "top_k": self.top_k,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return TeacherTargetProvenance(
            cache_key=hashlib.sha256(encoded).hexdigest(),
            teacher_fingerprint=self.controller.initial_fingerprint,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            prompt_template_fingerprint=self.formatter.fingerprint,
            feedback_fingerprint=feedback_fingerprint,
            continuation_fingerprint=continuation_fingerprint,
            residency=self.residency.value,
        )


def build_fixed_teacher_controller(
    student: Any,
    *,
    checkpoint_identity: str,
) -> TorchFixedTeacherController:
    """Deep-copy the initialized policy before any optimizer mutation."""
    try:
        teacher = copy.deepcopy(student)
    except Exception as exc:
        raise RuntimeError(
            "student cannot be deep-copied; load a separate fixed teacher for this quantization"
        ) from exc
    return TorchFixedTeacherController(teacher, checkpoint_identity=checkpoint_identity)


def _torch() -> Any:
    try:
        return __import__("torch")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for fixed-teacher scoring") from exc


__all__ = [
    "FileTeacherTargetCache",
    "TeacherTargetProvenance",
    "TransformersQuestionTeacherProvider",
    "build_fixed_teacher_controller",
]
