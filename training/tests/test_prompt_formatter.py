"""PromptFormatter normalizes the RLM's string, list, and mapping prompt shapes."""

from __future__ import annotations

import pytest

from rlm_train.models.transformers_runtime import GenerationConfig, PromptFormatter


def formatter() -> PromptFormatter:
    # messages() does not touch the tokenizer, so a placeholder is sufficient here.
    return PromptFormatter(object(), GenerationConfig())


def test_messages_accepts_rlm_message_list():
    history = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "solve it"},
    ]
    assert formatter().messages(history) == history


def test_messages_accepts_plain_string():
    result = formatter().messages("dense question")
    assert result[0]["role"] == "system"
    assert result[-1] == {"role": "user", "content": "dense question"}


def test_messages_accepts_mapping_with_messages():
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    assert formatter().messages(payload) == [{"role": "user", "content": "hi"}]


def test_messages_rejects_unsupported_type():
    with pytest.raises(TypeError):
        formatter().messages(123)


def test_messages_rejects_blank_content():
    with pytest.raises(ValueError, match="must not be blank"):
        formatter().messages([{"role": "user", "content": "   "}])
