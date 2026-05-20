from __future__ import annotations

"""Tiny dataset I/O shim for the screenshot-reproduced build scripts.

Original files use HuggingFace ``datasets`` and parquet.  The server training
project intentionally keeps dependencies light, so every script accepts either a
list of dicts, JSON, JSONL, or parquet when pandas/pyarrow are available.
"""

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

Row = dict[str, Any]
DatasetLike = list[Row]


def as_rows(dataset: Any) -> DatasetLike:
    if dataset is None:
        return []
    if isinstance(dataset, list):
        return [dict(row) for row in dataset]
    if hasattr(dataset, "to_list"):
        return [dict(row) for row in dataset.to_list()]
    return [dict(row) for row in dataset]


def load(path: str | Path) -> DatasetLike:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl" or path.suffix == ".parquet" and path.read_bytes()[:1] == b"{":
        rows: DatasetLike = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return [dict(row) for row in data["data"]]
            return [data]
        return [dict(row) for row in data]
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            return [dict(row) for row in pd.read_parquet(path).to_dict("records")]
        except Exception as exc:
            raise RuntimeError(f"cannot read parquet {path}; install pandas+pyarrow or use jsonl") from exc
    raise ValueError(f"unsupported dataset format: {path}")


def save(path: str | Path, dataset: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = as_rows(dataset)
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(rows).to_parquet(path, index=False)
            return path
        except Exception:
            # Keep smoke runnable: write JSONL fallback using the requested name.
            pass
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def dataset_from_list(rows: Sequence[Mapping[str, Any]]) -> DatasetLike:
    return [dict(row) for row in rows]


def concatenate_datasets(datasets: Iterable[Any]) -> DatasetLike:
    rows: DatasetLike = []
    for dataset in datasets:
        rows.extend(as_rows(dataset))
    return rows


__all__ = ["Row", "DatasetLike", "as_rows", "load", "save", "dataset_from_list", "concatenate_datasets"]
