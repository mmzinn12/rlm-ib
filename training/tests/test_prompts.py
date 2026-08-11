"""The RLM prompt asset files load and survive the RLM's str.format brace handling."""

from __future__ import annotations

from pathlib import Path

import pytest

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_FILES = ("rlm_root_system_prompt.md", "rlm_decomposition_fewshot.md")


def combined_prompt() -> str:
    return "\n\n".join((PROMPT_DIR / name).read_text(encoding="utf-8") for name in PROMPT_FILES)


def test_prompt_files_exist_and_mention_llm_query():
    text = combined_prompt()
    assert "llm_query(" in text
    assert 'answer["ready"]' in text
    assert "Write the specific helper question here." not in text


def test_escaped_prompt_is_format_safe():
    # The notebook escapes braces; the RLM then calls .format(custom_tools_section=...).
    escaped = combined_prompt().replace("{", "{{").replace("}", "}}")
    try:
        rendered = escaped.format(custom_tools_section="")
    except (KeyError, ValueError, IndexError) as exc:  # pragma: no cover - failure detail
        pytest.fail(f"escaped prompt is not format-safe: {exc}")
    # Braces round-trip back to single form for the model to read.
    assert '{"content": "", "ready": False}' in rendered
