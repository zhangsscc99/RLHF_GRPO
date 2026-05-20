from __future__ import annotations

"""End-to-end data cleaning/filtering stage from the screenshot pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional

import typer

from fastapply_pipeline.ast_tools import analyze_source
from .filter import filter_row
from .io import DatasetLike, as_rows, load, save
from .make import build as build_patch

app = typer.Typer(help="Clean FastApply examples by extension, AST/rules and token length.")


def approx_tokens(text: str) -> int:
    # Screenshots bucket/filter by tokenizer length.  Char/4 is the same cheap
    # approximation already visible in the reference images.
    return int(len(text) / 4) + 1


def clean_row(row: Dict[str, Any], max_seq_len: int = 32768, require_import: bool = True) -> Optional[Dict[str, Any]]:
    language = str(row.get("language") or "python")
    old_source = str(row.get("old_source") or "")
    new_source = str(row.get("new_source") or row.get("update") or "")
    diff = str(row.get("diff") or "")
    changes = row.get("changes") or []
    prompt_patch = build_patch(changes) if changes else diff
    if not old_source.strip() or not new_source.strip() or not diff.strip():
        return None
    if approx_tokens(old_source + new_source + prompt_patch) > max_seq_len:
        return None
    syntax = analyze_source(language, new_source)
    if not syntax.parse_ok:
        return None
    if not filter_row({**row, "old_source": old_source, "new_source": new_source, "diff": diff, "language": language}, require_import=require_import):
        return None
    out = dict(row)
    out["old_source"] = old_source
    out["new_source"] = new_source
    out["update"] = new_source
    out["tokens"] = approx_tokens(old_source + new_source + prompt_patch)
    out["syntax_checker"] = syntax.checker
    return out


def process(dataset: Any, max_seq_len: int = 32768, require_import: bool = True) -> DatasetLike:
    rows: DatasetLike = []
    for row in as_rows(dataset):
        cleaned = clean_row(row, max_seq_len=max_seq_len, require_import=require_import)
        if cleaned is not None:
            rows.append(cleaned)
    return rows


@app.command(name="clean")
def main(
    dataset_load_path: Path = typer.Option(..., help="dataset load path"),
    dataset_save_path: Optional[Path] = typer.Option(None, help="dataset save path"),
    max_seq_len: int = typer.Option(32768),
    require_import: bool = typer.Option(True),
):
    path_to_save = dataset_save_path or dataset_load_path
    rows = load(dataset_load_path)
    out = process(rows, max_seq_len=max_seq_len, require_import=require_import)
    save(path_to_save, out)
    typer.echo(f"cleaned {len(rows)} -> {len(out)} rows at {path_to_save}")


__all__ = ["approx_tokens", "clean_row", "process", "app"]
