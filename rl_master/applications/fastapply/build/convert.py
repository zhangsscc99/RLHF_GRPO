from __future__ import annotations

"""Convert cleaned FastApply rows into GRPO prompt/ground-truth rows."""

from pathlib import Path
from typing import Any, Dict, Optional

import typer

from fastapply_pipeline.templates import build_prompt, build_response
from .io import DatasetLike, as_rows, load, save

app = typer.Typer(help="Convert cleaned examples to GRPO prompt rows.")


def convert_item(item: Dict[str, Any]) -> Dict[str, Any]:
    language = str(item.get("language", "python"))
    source = str(item.get("old_source", ""))
    patch = str(item.get("patch") or item.get("diff") or "")
    update = str(item.get("update") or item.get("new_source") or "")
    prompt = build_prompt(language, source, patch)
    return {
        "data_source": "apply",
        "prompt": prompt,
        "ability": "fast_apply",
        "reward_model": {"style": "rule", "ground_truth": update},
        "extra_info": {"id": item.get("id"), "repo": item.get("repo"), "file": item.get("file"), "language": language},
        "response": build_response(update),
    }


def process(dataset: Any) -> DatasetLike:
    return [convert_item(row) for row in as_rows(dataset)]


@app.command(name="convert")
def main(
    dataset_load_path: Path = typer.Option(..., help="dataset load path"),
    dataset_save_path: Optional[Path] = typer.Option(None, help="dataset save path"),
):
    path_to_save = dataset_save_path or dataset_load_path
    out = process(load(dataset_load_path))
    save(path_to_save, out)
    typer.echo(f"converted {len(out)} rows -> {path_to_save}")


__all__ = ["convert_item", "process", "app"]
