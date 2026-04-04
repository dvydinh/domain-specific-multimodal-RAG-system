"""
Defensive JSON extraction utilities for LLM outputs.

LLMs frequently return malformed JSON: wrapped in markdown fences,
with trailing commas, or mixed with explanatory text. This module
provides a multi-layer parsing pipeline that gracefully degrades
instead of crashing the API with JSONDecodeError.

Additionally provides extract_text_content() to normalize LLM response
content that may be returned as str or list (thinking models).
"""

import json
import re
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_text_content(content: Union[str, list, None]) -> str:
    """
    Extract plain text from LLM response content.

    Some models (e.g. Gemma 4 with thinking mode) return content as a list
    of dicts: [{"type": "thinking", ...}, {"type": "text", "text": "..."}].
    Standard models return a plain string. This function normalizes both.

    Args:
        content: Raw response.content from LangChain LLM call.

    Returns:
        Extracted text string, or empty string if extraction fails.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") not in ("thinking",):
                    # Unknown block type — try to extract text anyway
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()

    # Fallback: convert to string
    return str(content).strip()


def extract_json(raw_text: Optional[str], fallback: Optional[dict] = None) -> dict:
    """
    Multi-layer JSON extraction from raw LLM output with runtime safety.

    Parsing pipeline:
      1. Direct json.loads()
      2. Strip markdown code fences (```json ... ```)
      3. Greedy extraction of outermost { ... }
      4. Return fallback dict

    Args:
        raw_text: Raw text from LLM that may contain JSON.
        fallback: Default dict to return if all parsing fails.

    Returns:
        Parsed dict, or fallback if all layers fail.
    """
    if fallback is None:
        fallback = {}

    if not raw_text or not isinstance(raw_text, str):
        logger.warning("Empty or non-string input to extract_json. Returning fallback.")
        return fallback

    text = raw_text.strip()
    if not text:
        return fallback

    # Layer 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Layer 2: Strip markdown fences (```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        try:
            content = fence_match.group(1).strip()
            if content:
                return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

    # Layer 3: Greedy extraction of outermost { ... }
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    # Layer 4: All parsing layers failed
    logger.warning(f"All JSON parsing layers failed. Raw text snippet: {text[:100]}...")
    return fallback
