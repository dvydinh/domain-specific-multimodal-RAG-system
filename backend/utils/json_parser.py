"""
Defensive JSON extraction utilities for LLM outputs.

LLMs frequently return malformed JSON: wrapped in markdown fences,
with trailing commas, or mixed with explanatory text. This module
provides a multi-layer parsing pipeline that gracefully degrades
instead of crashing the API with JSONDecodeError.
"""

import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_json(raw_text: str, fallback: Optional[dict] = None) -> dict:
    """
    Multi-layer JSON extraction from raw LLM output.
    
    Layer 1: Direct json.loads() (happy path)
    Layer 2: Regex extraction of {...} from markdown fences / surrounding text
    Layer 3: Return safe fallback dict
    
    Args:
        raw_text: Raw string from LLM response.
        fallback: Default dict if all parsing fails.
    
    Returns:
        Parsed dictionary, or fallback.
    """
    if fallback is None:
        fallback = {}

    text = raw_text.strip()

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
            return json.loads(fence_match.group(1).strip())
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
    logger.warning(f"All JSON parsing layers failed. Raw text: {text[:200]}...")
    return fallback
