from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass

from .ast_tools import analyze_source
from .patching import extract_updated_tag, normalize_code


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    exact: float
    similarity: float
    tag_format: float
    no_extra_prose: float
    syntax: float

    def to_json(self) -> dict[str, float]:
        return asdict(self)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_code(a), normalize_code(b)).ratio()


def score_fastapply_output(output: str, expected_source: str, language: str) -> RewardBreakdown:
    extracted = extract_updated_tag(output)
    tag_format = 1.0 if extracted is not None else 0.0
    candidate = extracted if extracted is not None else output
    exact = 1.0 if normalize_code(candidate) == normalize_code(expected_source) else 0.0
    similarity = _similarity(candidate, expected_source)
    stripped = output.strip()
    no_extra = 1.0 if (stripped.startswith("<updated>") and stripped.endswith("</updated>")) else 0.0
    syntax_info = analyze_source(language, candidate)
    syntax = 1.0 if syntax_info.parse_ok else 0.0
    # FastApply-style verifier: exact dominates, but partial rewards create GRPO signal.
    total = 0.55 * exact + 0.20 * similarity + 0.10 * tag_format + 0.05 * no_extra + 0.10 * syntax
    return RewardBreakdown(round(total, 6), exact, round(similarity, 6), tag_format, no_extra, syntax)
