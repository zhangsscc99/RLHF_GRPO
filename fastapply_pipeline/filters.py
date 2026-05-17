from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .ast_tools import analyze_source
from .patching import apply_search_replace, apply_structured_changes, normalize_code
from .schemas import ApplyExample


@dataclass
class FilterReport:
    kept: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    dedup_hashes: set[str] = field(default_factory=set)


def _fingerprint(ex: ApplyExample) -> str:
    material = "\n".join([ex.language, normalize_code(ex.old_source), normalize_code(ex.new_source)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def validate_patch_replays(ex: ApplyExample) -> tuple[bool, str]:
    if ex.change_kind == "search_replace":
        result = apply_search_replace(ex.old_source, ex.patch)
    elif ex.change_kind == "structured_changes":
        result = apply_structured_changes(ex.old_source, ex.patch)
    else:
        return True, "not replayed"
    if not result.ok:
        return False, result.error
    if normalize_code(result.updated) != normalize_code(ex.new_source):
        return False, "patch replay does not match new_source"
    return True, "ok"


def filter_examples(examples: list[ApplyExample], max_chars: int = 20_000) -> tuple[list[ApplyExample], FilterReport]:
    report = FilterReport()
    kept: list[ApplyExample] = []
    for ex in examples:
        if len(ex.old_source) + len(ex.patch) + len(ex.new_source) > max_chars:
            report.rejected[ex.id] = "oversized sample"
            continue
        ok, reason = validate_patch_replays(ex)
        if not ok:
            report.rejected[ex.id] = reason
            continue
        ast_report = analyze_source(ex.language, ex.new_source)
        if not ast_report.parse_ok:
            report.rejected[ex.id] = f"AST/syntax parse failed: {ast_report.error}"
            continue
        fp = _fingerprint(ex)
        if fp in report.dedup_hashes:
            report.rejected[ex.id] = "duplicate sample"
            continue
        report.dedup_hashes.add(fp)
        report.kept.append(ex.id)
        kept.append(ex)
    return kept, report
