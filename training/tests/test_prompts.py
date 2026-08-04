"""Verify the experiment-specific question-list prompt overlay.

Purpose:
    Protect optional and idempotent composition of training-only question guidance.
Implementation:
    Deterministic unit tests compose the constant overlay with a small base prompt.
Inputs:
    In-memory prompt strings and optional overlay values.
Outputs:
    Pytest assertions over composed prompt strings.
Example:
    Run ``pytest training/tests/test_prompts.py`` from the repository root.
"""

from rlm_train.prompts import QUESTION_LIST_PROMPT_OVERLAY, compose_training_system_prompt


def test_question_list_overlay_is_optional_and_idempotent():
    composed = compose_training_system_prompt("base")

    assert QUESTION_LIST_PROMPT_OVERLAY in composed
    assert compose_training_system_prompt(composed) == composed
    assert compose_training_system_prompt("base", question_list_overlay=None) == "base"
