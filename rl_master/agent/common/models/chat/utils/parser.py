from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Pattern, Set

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None


class RespondWithChoice:
    def __init__(self, pattern: str | Pattern[str], choices: Set[str]):
        self._pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self._choices = choices

    def __call__(self, response: str) -> Optional[str]:
        matched = re.search(self._pattern, response)
        if matched is None:
            return None
        choice = matched.group(1)
        return choice if choice in self._choices else None


class RespondWithJSON:
    def __init__(self, schema: Optional[Mapping[str, Any]] = None):
        self._schema = schema

    def __call__(self, response: str) -> Optional[Any]:
        def attempt(text: str):
            result = json.loads(text)
            if self._schema is not None and jsonschema is not None:
                jsonschema.validate(result, self._schema)
            return result
        candidates = [response.strip()]
        for left, right in [("{", "}"), ("[", "]")]:
            start, end = response.find(left), response.rfind(right)
            if 0 <= start < end:
                candidates.append(response[start:end + 1])
        for c in sorted(set(candidates), key=len, reverse=True):
            try:
                return attempt(c)
            except Exception:
                continue
        return None


class RespondWithPattern:
    def __init__(self, pattern: str | Pattern[str] = r"<answer>(.*?)</answer>", strip: bool = True):
        self._pattern = re.compile(pattern, re.DOTALL) if isinstance(pattern, str) else pattern
        self._strip = strip

    def __call__(self, response: str) -> Optional[str]:
        results = re.findall(self._pattern, response)
        if not results:
            return None
        result = results[-1]
        if self._strip:
            result = result.strip()
        return result if result not in {"", "."} else None
