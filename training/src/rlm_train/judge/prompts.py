"""Construct shared instructions for structured SDPO-RLM trajectory judges.

Purpose:
    Keep the subcall information-value definition consistent across judge providers.
Implementation:
    A fixed instruction block forbids outcome-contribution scoring and can be followed
    by optional task-specific evaluation instructions.
Inputs:
    An optional task-specific instruction string.
Outputs:
    A complete judge instruction string suitable for a system or developer prompt.
Example:
    ``prompt = build_judge_instructions("Check scientific support.")``
"""

from __future__ import annotations

import json

from rlm_train.judge.views import JudgeView

SUBCALL_INFORMATION_VALUE_INSTRUCTIONS = """
Evaluate each subcall or question by the significance of the information it revealed relative to
what was available before the call. Do not score it by whether it helped produce the final answer,
whether the final answer was correct, or whether the parent ultimately used the result. A subcall
can be valuable even when the final answer is wrong, and it can be low-value even when the final
answer is correct. Consider novelty, uncertainty reduction, evidentiary quality, redundancy, and
whether the result was misleading or invalid. Describe the concrete information revealed.

Relevance to the original task is a hard gate. If the helper question and its response do not help
resolve the original task, information significance and uncertainty reduction must be low or none,
regardless of fluency or factual correctness. An unrelated response must never receive high
significance or good evidence quality. Treat an unsupported response, a response contradicted by
the available evidence, or a factually false response as poor or mixed evidence and mark it
misleading when appropriate. Guidance for improvement must propose a helper question that advances
the original task rather than merely improving the unrelated question.
""".strip()


def build_judge_instructions(task_instructions: str = "") -> str:
    """Combine invariant subcall guidance with task-specific judge instructions.

    Args:
        task_instructions: Optional rubric text appended after the invariant guidance.

    Returns:
        The base information-value instructions, followed by the stripped custom text
        when provided.
    """
    sections = [SUBCALL_INFORMATION_VALUE_INSTRUCTIONS]
    if task_instructions:
        sections.append(task_instructions.strip())
    return "\n\n".join(sections)


def render_judge_view(view: JudgeView, *, instructions: str) -> tuple[str, str]:
    """Render only a pre-authorized typed view; never accept a full rollout here."""
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
