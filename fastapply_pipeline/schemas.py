from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ChangeKind = Literal["search_replace", "unified_diff", "structured_changes", "full_file"]


@dataclass(frozen=True)
class ApplyExample:
    id: str
    language: str
    old_source: str
    patch: str
    new_source: str
    change_kind: ChangeKind
    origin: str = "tiny_test_fixture"

    def to_json(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ApplyPromptRecord:
    id: str
    prompt: str
    response: str
    language: str
    change_kind: ChangeKind

    def to_json(self) -> dict[str, str]:
        return asdict(self)
