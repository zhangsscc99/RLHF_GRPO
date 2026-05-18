#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fastapply_pipeline.schemas import ApplyExample, ApplyPromptRecord
from fastapply_pipeline.templates import build_prompt, build_response
from fastapply_pipeline.test_data import tiny_apply_examples
from fastapply_pipeline.filters import filter_examples

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "fastapply_tiny_raw.jsonl"
PROMPT_PATH = DATA_DIR / "fastapply_tiny_prompts.jsonl"


def validate_example(ex: ApplyExample) -> None:
    assert ex.language
    assert ex.old_source.strip()
    assert ex.patch.strip()
    assert ex.new_source.strip()
    assert ex.new_source != ex.old_source
    assert "<updated>" not in ex.new_source


def to_prompt_record(ex: ApplyExample) -> ApplyPromptRecord:
    return ApplyPromptRecord(
        id=ex.id,
        prompt=build_prompt(ex.language, ex.old_source, ex.patch),
        response=build_response(ex.new_source),
        language=ex.language,
        change_kind=ex.change_kind,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    raw_examples = tiny_apply_examples()
    examples, filter_report = filter_examples(raw_examples)
    for ex in examples:
        validate_example(ex)
    records = [to_prompt_record(ex) for ex in examples]
    write_jsonl(RAW_PATH, [ex.to_json() for ex in examples])
    write_jsonl(PROMPT_PATH, [r.to_json() for r in records])
    print(json.dumps({
        "raw_path": str(RAW_PATH),
        "prompt_path": str(PROMPT_PATH),
        "num_examples": len(examples),
        "ids": [ex.id for ex in examples],
        "filter_report": {"kept": filter_report.kept, "rejected": filter_report.rejected},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
