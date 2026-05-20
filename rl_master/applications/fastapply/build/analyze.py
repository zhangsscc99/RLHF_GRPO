from __future__ import annotations

"""Length bucket analysis for cleaned data."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from .generate import approx_tokens
from .io import as_rows, load
from .make import build as build_patch

BUCKET_SIZE = 4096
MAX_SEQ_LEN = 16384

app = typer.Typer(help="Analyze token-length buckets.")


def analyze(dataset: Any) -> dict[str, int]:
    buckets: dict[str, int] = defaultdict(int)
    for item in as_rows(dataset):
        content = str(item.get("old_source", "")) + str(item.get("new_source", "")) + build_patch(item.get("changes") or [])
        tokens = int(item.get("tokens") or approx_tokens(content))
        if tokens > MAX_SEQ_LEN:
            name = "overflow"
        else:
            index = (tokens - 1) // BUCKET_SIZE
            start = index * BUCKET_SIZE + 1
            end = start + BUCKET_SIZE - 1
            name = f"{start}-{end}"
        buckets[name] += 1
    return dict(sorted(buckets.items()))


@app.command(name="analyze")
def main(path: Path = typer.Option(Path("."), help="dataset path")):
    rows = load(path)
    for name, count in analyze(rows).items():
        typer.echo(f"{name}: {count}")


__all__ = ["analyze", "app", "BUCKET_SIZE", "MAX_SEQ_LEN"]
