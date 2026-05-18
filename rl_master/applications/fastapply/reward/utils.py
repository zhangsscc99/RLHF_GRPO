from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class Hunk:
    old: str
    new: str


class Parser:
    _pattern = re.compile(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

    def __call__(self, old_source: str, new_source: str, diff: str) -> Optional[List[Dict[str, str]]]:
        def build(hunk: Hunk, version: Literal["old", "new"]) -> str:
            marker = "old" if version == "old" else "new"
            return "".join(
                line[1:] for line in hunk.old.splitlines(keepends=True)
                if not line.startswith("+")
            ) if marker == "old" else "".join(
                line[1:] for line in hunk.new.splitlines(keepends=True)
                if not line.startswith("-")
            )

        hunks: List[str] = []
        current: List[str] = []
        for line in diff.splitlines(keepends=True):
            if self._pattern.match(line):
                if current:
                    hunks.append("".join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            hunks.append("".join(current))
        if not hunks:
            return None
        result = []
        for h in hunks:
            old_lines = []
            new_lines = []
            for line in h.splitlines(keepends=True):
                if line.startswith("@@"):
                    continue
                if line.startswith("-"):
                    old_lines.append(line)
                elif line.startswith("+"):
                    new_lines.append(line)
                else:
                    old_lines.append(" " + line if not line.startswith(" ") else line)
                    new_lines.append(" " + line if not line.startswith(" ") else line)
            old = "".join(l[1:] for l in old_lines)
            new = "".join(l[1:] for l in new_lines)
            if old.strip() or new.strip():
                result.append({"old": old, "new": new, "hunk": h})
        return result


def build(changes: List[Dict[str, str]]) -> str:
    result = []
    for change in changes:
        content = "<hunk>\n"
        content += "<<<<<<< SEARCH\n"
        content += change["old"].rstrip("\n") + "\n"
        content += "=======\n"
        content += change["new"].rstrip("\n") + "\n"
        content += ">>>>>>> REPLACE\n"
        content += "</hunk>"
        result.append(content)
    return "\n".join(result)


def generate_diff_view(reference: str, candidate: str, language: Literal["python", "java", "javascript", "typescript"] = "python", use_conflict_format: bool = False) -> str:
    ext = {"python": ".py", "java": ".java", "javascript": ".js", "typescript": ".ts"}.get(language, ".txt")
    diff = "".join(difflib.unified_diff(
        reference.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile=f"reference{ext}",
        tofile=f"candidate{ext}",
        n=3,
    ))
    if use_conflict_format:
        parser = Parser()
        changes = parser(reference, candidate, diff)
        return build(changes or [])
    return diff


def compare(reference: str, candidate: str) -> bool:
    reference = "\n".join(line for line in reference.splitlines() if line.strip())
    candidate = "\n".join(line for line in candidate.splitlines() if line.strip())
    diff = list(difflib.Differ().compare(reference.splitlines(), candidate.splitlines()))
    additions = sum(1 for line in diff if line.startswith("+ "))
    deletions = sum(1 for line in diff if line.startswith("- "))
    return additions == 0 and deletions == 0
