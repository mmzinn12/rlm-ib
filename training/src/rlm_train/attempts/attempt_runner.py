"""Run complete recursive-language-model attempts through the shared student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rlm.clients.base_lm import BaseLM
from rlm.core.rlm import RLM
from rlm.core.types import RLMChatCompletion

from rlm_train.attempts.attempt_records import AnnotatedAttempt
from rlm_train.attempts.record_attempt import AttemptRecorder
from rlm_train.datasets.records import DatasetRecord, require_question_context
from rlm_train.spec.rollout import RolloutSpec

AttemptMode = Literal["training", "evaluation"]


@dataclass(frozen=True)
class AttemptRequest:
    task_id: str
    public_task: dict[str, Any]
    private_reference: Any | None = None
    mode: AttemptMode = "training"


@dataclass(frozen=True)
class AttemptResult:
    completion: RLMChatCompletion
    attempt: AnnotatedAttempt


class AttemptRunner(Protocol):
    def run(self, request: AttemptRequest) -> AttemptResult: ...

    def run_many(
        self,
        record: DatasetRecord,
        *,
        count: int,
        mode: AttemptMode = "training",
    ) -> tuple[AnnotatedAttempt, ...]: ...


class RLMAttemptRunner:
    """Run root, plain, and recursive generations through one student client."""

    def __init__(
        self,
        *,
        student_client: BaseLM,
        student_id: str,
        spec: RolloutSpec,
        backend: str = "openai",
        environment_kwargs: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.student_client = student_client
        self.student_id = student_id
        self.spec = spec
        self.backend = backend
        self.environment_kwargs = dict(environment_kwargs or {})
        self.provenance = dict(provenance or {})

    def run(self, request: AttemptRequest) -> AttemptResult:
        recorder = AttemptRecorder(
            task_id=request.task_id,
            public_task=request.public_task,
            private_reference=request.private_reference,
            student={
                # Schema version 1 intentionally retains this persisted field name.
                "policy_owner": self.student_id,
                "model_id": self.student_client.model_name,
            },
            mode=request.mode,
            provenance=self.provenance,
        )
        rlm = RLM(
            backend=self.backend,
            backend_kwargs={"model_name": self.student_client.model_name},
            client=self.student_client,
            environment=self.spec.environment,
            environment_kwargs=self.environment_kwargs,
            max_depth=self.spec.max_depth,
            max_iterations=self.spec.max_iterations,
            max_concurrent_subcalls=self.spec.max_concurrent_subcalls,
            persistent=self.spec.persistent,
            custom_system_prompt=self.spec.system_prompt,
            sampling_args=self.spec.sampling,
            sub_sampling_args=self.spec.subcall_sampling,
            observer=recorder,
            policy_owner=self.student_id,
        )
        try:
            question, context = require_question_context(request.public_task)
            completion = rlm.completion(context, root_prompt=question)
        finally:
            rlm.close()
        attempt = recorder.create_attempt(result={"final_answer": completion.response})
        return AttemptResult(completion=completion, attempt=attempt)

    def run_many(
        self,
        record: DatasetRecord,
        *,
        count: int,
        mode: AttemptMode = "training",
    ) -> tuple[AnnotatedAttempt, ...]:
        if count <= 0:
            raise ValueError("attempt count must be positive")
        request = AttemptRequest(
            task_id=record.record_id,
            public_task=record.public_task,
            private_reference=record.verifier_data,
            mode=mode,
        )
        return tuple(self.run(request).attempt for _ in range(count))


__all__ = [
    "AttemptMode",
    "AttemptRequest",
    "AttemptResult",
    "AttemptRunner",
    "RLMAttemptRunner",
]
