"""PromptFormatter normalizes the RLM's string, list, and mapping prompt shapes."""

from __future__ import annotations

import pytest

from rlm_train.models.transformers_runtime import (
    GenerationConfig,
    PromptFormatter,
    decode_token_offsets,
)


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


def test_messages_allows_a_single_blank_turn_among_content():
    history = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": "next"},
    ]
    assert formatter().messages(history) == history


def test_messages_rejects_all_blank_messages():
    with pytest.raises(ValueError, match="non-blank"):
        formatter().messages([{"role": "user", "content": "   "}])


class AsciiTokenizer:
    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "abc"[: len(ids)]


class ByteSplitTokenizer:
    # Simulates a multi-byte character split across the 2nd and 3rd tokens.
    TABLE = {(1,): "a", (1, 2): "a\ufffd", (1, 2, 3): "a\u00e9"}

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return self.TABLE[tuple(ids)]


def test_decode_token_offsets_ascii_is_contiguous():
    offsets = decode_token_offsets(AsciiTokenizer(), (1, 2, 3), expected_text="abc")
    assert offsets == ((0, 1), (1, 2), (2, 3))


def test_decode_token_offsets_defers_incomplete_multibyte_char():
    offsets = decode_token_offsets(ByteSplitTokenizer(), (1, 2, 3), expected_text="a\u00e9")
    assert offsets == ((0, 1), (1, 1), (1, 2))
