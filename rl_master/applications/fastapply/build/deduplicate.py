from __future__ import annotations

"""Deduplicate examples by update/content signature.

The screenshots use datasketch MinHash over AST-normalized code.  This adaptation
provides the same ``Signature``/``deduplicate`` interface with a deterministic
hash fallback; if datasketch is installed it can be swapped in without changing
callers.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import typer

from .io import DatasetLike, as_rows, load, save

app = typer.Typer(help="Deduplicate similar update snippets.")


class Signature:
    def __init__(self, language: Literal["python", "java"] | str = "python"):
        self.language = str(language)

    def __call__(self, code: str) -> str:
        return self.serialize(self.normalize(code))

    def normalize(self, code: str) -> dict[str, Any]:
        tokens = re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|[-+*/%{}()[\].,;:]", code)
        normalized: list[str] = []
        for token in tokens:
            if re.match(r"^[A-Za-z_]\w*$", token) and token not in {"def", "class", "import", "from", "return", "if", "else", "for", "while", "public", "private", "protected", "static", "void", "int", "String"}:
                normalized.append("ID")
            elif token.isdigit():
                normalized.append("NUM")
            else:
                normalized.append(token)
        return {"type": "tokens", "children": normalized}

    def serialize(self, node: dict[str, Any]) -> str:
        payload = " ".join(str(x) for x in node.get("children", []))
        return hashlib.sha1(f"{self.language}:{payload}".encode("utf-8")).hexdigest()


def deduplicate(dataset: Any) -> DatasetLike:
    similarity_indices: dict[str, set[str]] = {}
    result: DatasetLike = []
    for item in as_rows(dataset):
        language = str(item.get("language", "python"))
        signature = Signature(language)(str(item.get("update") or item.get("new_source") or ""))
        bucket = similarity_indices.setdefault(language, set())
        if signature in bucket:
            continue
        bucket.add(signature)
        result.append(item)
    return result


@app.command(name="deduplicate")
def main(
    dataset_load_path: Path = typer.Option(..., help="dataset load path"),
    dataset_save_path: Optional[Path] = typer.Option(None, help="dataset save path"),
):
    path_to_save = dataset_save_path or dataset_load_path
    out = deduplicate(load(dataset_load_path))
    save(path_to_save, out)
    typer.echo(f"deduplicated -> {len(out)} rows at {path_to_save}")


__all__ = ["Signature", "deduplicate", "app"]
