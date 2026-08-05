"""Adapter around the repository's one canonical full-RLM execution engine."""

from __future__ import annotations

from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.core.rlm import RLM

from rlm_train.rollouts.protocol import RolloutRequest, RolloutResult
from rlm_train.rollouts.recorder import RolloutRecorder
from rlm_train.spec.rollout import RolloutSpec


class RLMRolloutEngine:
    """Run root and recursive generations through one injected student policy."""

    def __init__(
        self,
        *,
        policy: BaseLM,
        policy_owner: str,
        spec: RolloutSpec,
        backend: str = "openai",
        environment_kwargs: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.policy_owner = policy_owner
        self.spec = spec
        self.backend = backend
        self.environment_kwargs = dict(environment_kwargs or {})
        self.provenance = dict(provenance or {})

    def execute(self, request: RolloutRequest) -> RolloutResult:
        recorder = RolloutRecorder(
            task_id=request.task_id,
            public_task=request.public_task,
            private_reference=request.private_reference,
            policy={
                "policy_owner": self.policy_owner,
                "model_id": self.policy.model_name,
            },
            mode=request.mode,
            provenance=self.provenance,
        )
        rlm = RLM(
            backend=self.backend,
            backend_kwargs={"model_name": self.policy.model_name},
            client=self.policy,
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
            policy_owner=self.policy_owner,
        )
        try:
            prompt: str | dict[str, Any]
            prompt = request.public_task.get("prompt", request.public_task)
            completion = rlm.completion(prompt)
        finally:
            rlm.close()
        rollout = recorder.build(result={"final_answer": completion.response})
        return RolloutResult(completion=completion, rollout=rollout)


__all__ = ["RLMRolloutEngine"]
