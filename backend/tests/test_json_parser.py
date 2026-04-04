"""
Unit tests for JSON parser and content extraction utilities.

Tests cover:
  - extract_text_content: normalizes str, list (thinking mode), and None inputs
  - extract_json: multi-layer parsing (direct, markdown fence, regex, fallback)
"""

import pytest
from backend.utils.json_parser import extract_json, extract_text_content


class TestExtractTextContent:
    """Tests for the model-agnostic content extractor."""

    def test_plain_string(self):
        assert extract_text_content("hello world") == "hello world"

    def test_string_with_whitespace(self):
        assert extract_text_content("  hello  ") == "hello"

    def test_none_returns_empty(self):
        assert extract_text_content(None) == ""

    def test_empty_string(self):
        assert extract_text_content("") == ""

    def test_gemma4_thinking_mode(self):
        """Gemma 4 returns list with thinking + text blocks."""
        content = [
            {"type": "thinking", "text": "Let me think about this..."},
            {"type": "text", "text": "The answer is 42."},
        ]
        result = extract_text_content(content)
        assert result == "The answer is 42."
        assert "thinking" not in result.lower() or "think" not in result

    def test_multiple_text_blocks(self):
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        assert extract_text_content(content) == "Part 1\nPart 2"

    def test_list_of_strings(self):
        content = ["hello", "world"]
        assert extract_text_content(content) == "hello\nworld"

    def test_empty_list(self):
        assert extract_text_content([]) == ""

    def test_thinking_only_returns_empty(self):
        content = [{"type": "thinking", "text": "hmm..."}]
        assert extract_text_content(content) == ""


class TestExtractJson:
    """Tests for the multi-layer JSON parser."""

    def test_direct_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"recipes": []}\n```'
        result = extract_json(raw)
        assert result == {"recipes": []}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the data: {"name": "test"} and some more text'
        result = extract_json(raw)
        assert result == {"name": "test"}

    def test_empty_input_returns_fallback(self):
        assert extract_json("") == {}
        assert extract_json(None) == {}
        assert extract_json("  ") == {}

    def test_custom_fallback(self):
        fallback = {"recipes": []}
        result = extract_json("not json at all", fallback=fallback)
        assert result == fallback

    def test_invalid_json_returns_fallback(self):
        result = extract_json("This is just plain text with no JSON")
        assert result == {}

    def test_nested_json(self):
        raw = '{"recipes": [{"name": "Ramen", "ingredients": ["noodles"]}]}'
        result = extract_json(raw)
        assert result["recipes"][0]["name"] == "Ramen"

    def test_markdown_fence_without_json_label(self):
        raw = '```\n{"key": "val"}\n```'
        result = extract_json(raw)
        assert result == {"key": "val"}
