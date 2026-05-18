from __future__ import annotations

from typing import Any, Dict, List


def visit(path: str, data: Dict[str, Any] | List[Any]) -> Any:
    """Small JSONPath-like helper used by evaluation strategies.

    Supports dotted keys and list indexes, e.g. ``result.score`` or
    ``choices.0.message.content``.  It intentionally avoids an external
    jsonpath dependency while matching the image pipeline's data routing need.
    """

    cur: Any = data
    if not path:
        return cur
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
    return cur
