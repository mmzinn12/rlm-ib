"""Create the instructions and authorized payload sent to a feedback judge."""

from __future__ import annotations

import json

from rlm_train.feedback.feedback_views import FeedbackView

SUBCALL_INFORMATION_VALUE_INSTRUCTIONS = """
## AUTHORITATIVE TASK DEFINITION

- OBJECTIVE: The value of `task.question`.
- SUPPORTING EVIDENCE: The value of `task.context`.
- IMPORTANT: Context passages are evidence, not separate tasks.
- Ignore a passage’s topic unless it helps answer the question.

## EVALUATION TARGET

Evaluate only:
1. The helper question.
2. The helper response.
3. Information explicitly contained in that response.

Do not credit facts merely present in the task context.
Do not infer facts the helper response did not state.

## DECISION PROCEDURE

1. Restate the objective from `task.question`.
2. Identify the factual claims made by the helper response.
3. Compare those claims with the supplied context.
4. Determine whether they reduce uncertainty about the objective.
5. Mark unsupported or contradicted claims as misleading.
6. Populate the output schema.

## REQUIRED

- Treat `task.question` as the original task.
- Distinguish the context corpus from the task objective.
- Report only information actually revealed by the helper response.
- Check factual claims against the provided context.

## FORBIDDEN

- Do not treat the first context passage as the original task.
- Do not reward a helper response merely because its conclusion is correct.
- Do not supply missing facts from your own knowledge.
- Do not claim the response revealed facts that appear only in context.
""".strip()


def create_judge_instructions(task_instructions: str = "") -> str:
    """Combine invariant information-value guidance with a task-specific contract."""
    sections = [SUBCALL_INFORMATION_VALUE_INSTRUCTIONS]
    if task_instructions:
        sections.append(task_instructions.strip())
    return "\n\n".join(sections)


def render_feedback_view(view: FeedbackView, *, instructions: str) -> tuple[str, str]:
    """Serialize only a pre-authorized feedback view for the judge request."""
    if not instructions.strip():
        raise ValueError("judge instructions must not be blank")
    payload = json.dumps(
        view.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return instructions, payload


__all__ = [
    "SUBCALL_INFORMATION_VALUE_INSTRUCTIONS",
    "create_judge_instructions",
    "render_feedback_view",
]
