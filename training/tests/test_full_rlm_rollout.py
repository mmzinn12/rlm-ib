"""Full canonical rollout instrumentation across plain and recursive subcalls."""

import threading

import pytest
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

from rlm_train.attempts import AttemptRequest, RLMAttemptRunner
from rlm_train.spec import RolloutSpec


class ExactTokenFakePolicy(BaseLM):
    def __init__(self) -> None:
        super().__init__("exact-token-fake")
        self.responses = [
            (
                "```repl\n"
                "plain = llm_query('plain question')\n"
                "child = rlm_query('recursive question')\n"
                "answer['content'] = plain + '|' + child\n"
                "answer['ready'] = True\n"
                "```"
            ),
            "plain response",
            ("```repl\nanswer['content'] = 'recursive response'\nanswer['ready'] = True\n```"),
        ]
        self.lock = threading.Lock()
        self.local = threading.local()
        self.call_count = 0
        self.prompts = []

    def completion(self, prompt):
        with self.lock:
            self.prompts.append(prompt)
            response = self.responses.pop(0)
            self.call_count += 1
        self.local.response = response
        return response

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary({self.model_name: ModelUsageSummary(self.call_count, 0, 0, 0.0)})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(1, 0, 0, 0.0)

    def get_last_generation(self):
        response = self.local.response
        return {
            "prompt_token_ids": (9001,),
            "token_ids": tuple(range(len(response))),
            "token_offsets": tuple((index, index + 1) for index in range(len(response))),
            "prompt_token_count": 1,
            "policy_owner": "student:exact",
        }


def test_full_rlm_attempt_records_root_plain_recursive_execution_and_final_answer():
    policy = ExactTokenFakePolicy()
    runner = RLMAttemptRunner(
        student_client=policy,
        student_id="student:exact",
        spec=RolloutSpec(max_depth=2, max_iterations=2),
    )

    result = runner.run(
        AttemptRequest(
            task_id="task",
            public_task={"question": "solve the task", "context": "supporting evidence"},
            private_reference={"answer": "verifier secret"},
        )
    )

    event_types = [event["event_type"] for event in result.attempt.execution.events]
    assert result.completion.response == "plain response|recursive response"
    assert "Answer the following: solve the task" in str(policy.prompts[0])
    root = next(node for node in result.attempt.execution.nodes if node.role.value == "root")
    assert root.prompt == "supporting evidence"
    assert "code_execution_started" in event_types
    assert "plain_subcall_started" in event_types
    assert "recursive_subcall_started" in event_types
    assert "final_answer_submitted" in event_types
    assert {node.role.value for node in result.attempt.execution.nodes} == {
        "root",
        "plain_subcall",
        "recursive_subcall",
    }
    assert {node.policy_owner for node in result.attempt.execution.nodes} == {"student:exact"}
    assert len(result.attempt.annotations.generations) == 3
    assert all(
        generation.prompt_token_ids == (9001,)
        for generation in result.attempt.annotations.generations
    )
    assert all(generation.token_ids for generation in result.attempt.annotations.generations)
    assert "verifier secret" not in result.attempt.canonical_json()
    assert result.attempt.task.private_reference_fingerprint is not None


def test_attempt_runner_closes_rlm_when_execution_raises(monkeypatch):
    closed = False

    class FailingRLM:
        def __init__(self, **kwargs):
            del kwargs

        def completion(self, context, *, root_prompt):
            del context, root_prompt
            raise RuntimeError("generation failed")

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr("rlm_train.attempts.attempt_runner.RLM", FailingRLM)
    runner = RLMAttemptRunner(
        student_client=ExactTokenFakePolicy(),
        student_id="student:exact",
        spec=RolloutSpec(),
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        runner.run(
            AttemptRequest(
                task_id="task",
                public_task={"question": "solve", "context": "evidence"},
            )
        )

    assert closed is True


def test_attempt_runner_rejects_non_positive_attempt_count():
    runner = RLMAttemptRunner(
        student_client=ExactTokenFakePolicy(),
        student_id="student:exact",
        spec=RolloutSpec(),
    )

    with pytest.raises(ValueError, match="attempt count must be positive"):
        runner.run_many(object(), count=0)
