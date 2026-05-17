from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ApplyResult:
    ok: bool
    updated: str
    error: str = ""


def normalize_code(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def make_unified_diff(old: str, new: str, filename: str = "file") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )


def make_search_replace(old_fragment: str, new_fragment: str) -> str:
    return f"<<<<<<< SEARCH\n{old_fragment.rstrip()}\n=======\n{new_fragment.rstrip()}\n>>>>>>> REPLACE"


def parse_search_replace_hunks(patch: str) -> list[tuple[str, str]]:
    pat = re.compile(r"<{7} SEARCH\n(?P<old>[\s\S]*?)\n={7}\n(?P<new>[\s\S]*?)\n>{7} REPLACE")
    return [(m.group("old"), m.group("new")) for m in pat.finditer(patch)]


def apply_search_replace(source: str, patch: str) -> ApplyResult:
    updated = source
    hunks = parse_search_replace_hunks(patch)
    if not hunks:
        return ApplyResult(False, source, "no SEARCH/REPLACE hunks")
    for old, new in hunks:
        count = updated.count(old)
        if count != 1:
            return ApplyResult(False, updated, f"SEARCH fragment count={count}, expected 1")
        updated = updated.replace(old, new, 1)
    return ApplyResult(True, updated)


def apply_structured_changes(source: str, patch: str) -> ApplyResult:
    try:
        obj = json.loads(patch)
    except json.JSONDecodeError as exc:
        return ApplyResult(False, source, f"invalid structured JSON: {exc}")
    updated = source
    changes = obj.get("changes")
    if not isinstance(changes, list):
        return ApplyResult(False, source, "missing changes list")
    for ch in changes:
        kind = ch.get("kind")
        if kind == "replace":
            old, new = ch.get("old", ""), ch.get("new", "")
            if updated.count(old) != 1:
                return ApplyResult(False, updated, "replace old fragment is ambiguous or missing")
            updated = updated.replace(old, new, 1)
        elif kind == "insert_after":
            anchor, text = ch.get("anchor", ""), ch.get("text", "")
            if updated.count(anchor) != 1:
                return ApplyResult(False, updated, "insert anchor is ambiguous or missing")
            updated = updated.replace(anchor, anchor + text, 1)
        elif kind == "delete":
            old = ch.get("old", "")
            if updated.count(old) != 1:
                return ApplyResult(False, updated, "delete old fragment is ambiguous or missing")
            updated = updated.replace(old, "", 1)
        else:
            return ApplyResult(False, updated, f"unsupported change kind: {kind}")
    return ApplyResult(True, updated)


def extract_updated_tag(text: str) -> str | None:
    m = re.search(r"<updated>\s*\n?([\s\S]*?)\n?\s*</updated>", text)
    return m.group(1) if m else None
