# utils/json_utils.py — safe JSON parsing helpers

import json
import re


def safe_parse_json(text: str) -> dict | list | None:
    """
    Parse JSON from a model response that may contain markdown fences or
    leading/trailing prose.

    Tries three strategies in order:
      1. Direct parse (model responded cleanly)
      2. Strip ```json ... ``` fences
      3. Extract the first {...} or [...] block via regex
    """
    if not text or not text.strip():
        return None

    # Strategy 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\n?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract first JSON block
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None
