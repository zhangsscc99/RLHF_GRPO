from __future__ import annotations

import json
import re
from typing import Any, Optional


def _remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def get_last_boxed_string(content: str) -> Optional[str]:
    left = "\\boxed{"
    idx = content.rfind(left)
    if idx < 0:
        return None
    i = idx + len(left)
    depth = 1
    while i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[idx:i + 1]
        i += 1
    return None


def extract_boxed_content(content: str) -> Optional[str]:
    boxed = get_last_boxed_string(content)
    return _remove_boxed(boxed) if boxed else None


def extract_tagged_content(content: str, tag: str = "answer") -> Optional[str]:
    if not isinstance(content, str):
        return None
    results = re.findall(fr"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    if not results:
        return None
    result = results[-1].strip()
    return result if result not in {"", "."} else None


def extract_json(content: str) -> Optional[Any]:
    try:
        return json.loads(content)
    except Exception:
        pass
    # extract longest likely JSON object/list
    candidates = []
    for left, right in [("{", "}"), ("[", "]")]:
        start = content.find(left)
        end = content.rfind(right)
        if 0 <= start < end:
            candidates.append(content[start:end + 1])
    for snippet in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None
