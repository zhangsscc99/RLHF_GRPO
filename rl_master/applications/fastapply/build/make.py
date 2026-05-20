from __future__ import annotations

"""Generate/update ``<updated>`` snippets for parsed FastApply examples.

The screenshot implementation asks an LLM to produce an update snippet from
SEARCH/REPLACE hunks.  For a runnable reproduction, ``Generator`` keeps the same
shape but uses a deterministic path when ``new_source`` is already present; this
is what the tiny smoke dataset uses.  A caller can plug a model object by passing
an async callable with the same ``Generator`` interface.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import typer

from fastapply_pipeline.patching import apply_search_replace
from .io import DatasetLike, as_rows, dataset_from_list, load, save

app = typer.Typer(help="Generate update snippet by LLM or deterministic fallback.")


def build(changes: list[dict[str, str]]) -> str:
    result: list[str] = []
    for change in changes:
        content = "<hunk>\n"
        content += "<<<<<<< SEARCH\n"
        content += change.get("old", "") + "\n"
        content += "=======\n"
        content += change.get("new", "") + "\n"
        content += ">>>>>>> REPLACE\n"
        content += "</hunk>"
        result.append(content)
    return "\n".join(result)


def build_search_replace_patch(changes: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for change in changes:
        blocks.append("<<<<<<< SEARCH\n{}\n=======\n{}\n>>>>>>> REPLACE".format(change.get("old", ""), change.get("new", "")))
    return "\n".join(blocks)


class Generator:
    def __init__(self, model: Optional[Any] = None):
        self._model = model

    async def __call__(self, item: Dict[str, Any]) -> Optional[str]:
        if item.get("new_source"):
            return str(item["new_source"])
        changes = item.get("changes") or []
        if item.get("old_source") and changes:
            result = apply_search_replace(str(item["old_source"]), build_search_replace_patch(changes))
            if result.ok:
                return result.updated
        if self._model is not None:
            data = {"language": item.get("language"), "source": item.get("old_source"), "patch": build(changes)}
            maybe = self._model(data)
            if asyncio.iscoroutine(maybe):
                maybe = await maybe
            if maybe is not None:
                return str(maybe)
        return None


async def process(model: Optional[Any], dataset: Any) -> DatasetLike:
    generate = Generator(model)
    semaphore = asyncio.Semaphore(32)

    async def transform(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with semaphore:
            update = await generate(sample)
            if update is None:
                return None
            row = dict(sample)
            row["update"] = update
            return row

    tasks = [transform(row) for row in as_rows(dataset)]
    results = await asyncio.gather(*tasks)
    return [row for row in results if row is not None]


@app.command(name="generate")
def main(
    dataset_load_path: Path = typer.Option(..., help="dataset load path"),
    dataset_save_path: Path = typer.Option(..., help="dataset save path"),
    start: int = typer.Option(0),
    end: int = typer.Option(1024),
):
    rows = load(dataset_load_path)[start:end]
    out = asyncio.run(process(None, rows))
    save(dataset_save_path, out)
    typer.echo(f"generated {len(out)} rows -> {dataset_save_path}")


__all__ = ["build", "build_search_replace_patch", "Generator", "process", "app"]
