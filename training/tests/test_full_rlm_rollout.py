"""Full canonical rollout instrumentation across plain and recursive subcalls."""

import threading

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

from rlm_train.rollouts import RLMRolloutEngine, RolloutRequest
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

    def completion(self, prompt):
        del prompt
        with self.lock:
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


def test_full_rlm_rollout_records_root_plain_recursive_execution_and_final_answer():
    engine = RLMRolloutEngine(
        policy=ExactTokenFakePolicy(),
        policy_owner="student:exact",
        spec=RolloutSpec(max_depth=2, max_iterations=2),
    )

    result = engine.execute(
        RolloutRequest(task_id="task", public_task={"prompt": "solve the task"})
    )

    event_types = [event["event_type"] for event in result.rollout.execution.events]
    assert result.completion.response == "plain response|recursive response"
    assert "code_execution_started" in event_types
    assert "plain_subcall_started" in event_types
    assert "recursive_subcall_started" in event_types
    assert "final_answer_submitted" in event_types
    assert {node.role.value for node in result.rollout.execution.nodes} == {
        "root",
        "plain_subcall",
        "recursive_subcall",
    }
    assert {node.policy_owner for node in result.rollout.execution.nodes} == {"student:exact"}
    assert len(result.rollout.annotations.generations) == 3
    assert all(generation.token_ids for generation in result.rollout.annotations.generations)
