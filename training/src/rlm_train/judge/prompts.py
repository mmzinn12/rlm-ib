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

SUBCALL_INFORMATION_VALUE_INSTRUCTIONS = """
Evaluate each subcall or question by the significance of the information it revealed relative to
what was available before the call. Do not score it by whether it helped produce the final answer,
whether the final answer was correct, or whether the parent ultimately used the result. A subcall
can be valuable even when the final answer is wrong, and it can be low-value even when the final
answer is correct. Consider novelty, uncertainty reduction, evidentiary quality, redundancy, and
whether the result was misleading or invalid. Describe the concrete information revealed.
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
