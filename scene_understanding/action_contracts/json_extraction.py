"""Robust JSON extraction for LLM-generated contract fragments.

Do not use greedy regexes such as /\[[\s\S]*\]/ for action grounding output.
Those can slice the middle of an object when the first field is an array, causing
errors like: ``Claude returned non-JSON: [[590, 150], ...], "depth_trace_m"``.
This module extracts the first balanced JSON object or array while respecting
strings and escapes.
"""
from __future__ import annotations

import json
from typing import Any, Tuple


class JsonExtractionError(ValueError):
    """Raised when no complete JSON value can be extracted."""


def extract_first_json_value(text: str) -> Any:
    """Return the first balanced JSON object or array in ``text``.

    The scanner is string-aware and does not get confused by braces/brackets in
    quoted strings. It is safe for UI/agent responses that contain prose before
    or after the machine-readable payload.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    start, opener = _find_first_json_start(text)
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise JsonExtractionError("Found JSON start but no balanced closing delimiter")


def extract_first_json_object(text: str) -> dict:
    """Return the first JSON object from ``text``.

    Arrays are accepted by ``extract_first_json_value`` but rejected here because
    grounding responses should be objects with named fields. This prevents naked
    coordinate arrays from accidentally being treated as complete contracts.
    """

    value = extract_first_json_value(text)
    if not isinstance(value, dict):
        raise JsonExtractionError("Expected a JSON object, not a naked array/value")
    return value


def _find_first_json_start(text: str) -> Tuple[int, str]:
    obj = text.find("{")
    arr = text.find("[")
    starts = [(idx, ch) for idx, ch in ((obj, "{"), (arr, "[")) if idx >= 0]
    if not starts:
        raise JsonExtractionError("No JSON object or array found")
    return min(starts, key=lambda pair: pair[0])
