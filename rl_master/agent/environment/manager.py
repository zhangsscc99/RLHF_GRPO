from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DataItem:
    language: str
    source: str
    patch: str
    reference: str
    metadata: Dict[str, Any]


class GlobalDatasetManager:
    """Tiny stand-in for the RL-master/Ray dataset manager shown in the images.

    It supports JSONL/JSON locally and keeps the same interface shape used by the
    FastApply environment: register a split, then request an item by seed.
    """

    def __init__(self, path: str | Path = "", split: str = "train"):
        self.path = Path(path) if path else None
        self.split = split
        self._items: List[DataItem] = []
        if self.path and self.path.exists():
            self._items = list(self._load(self.path))

    @classmethod
    def options(cls, **kwargs):  # Ray-compatible no-op used by screenshots.
        return cls

    @classmethod
    def remote(cls, *args, **kwargs):  # Ray-compatible constructor.
        return cls(*args, **kwargs)

    def register(self, dataset: "GlobalDatasetManager") -> None:
        self._items = list(dataset._items)

    def get_data_item(self, seed: int = 0) -> Optional[DataItem]:
        if not self._items:
            return None
        return self._items[seed % len(self._items)]

    def _load(self, path: Path) -> Iterable[DataItem]:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield self._coerce(json.loads(line))
        elif path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else obj.get(self.split, obj.get("data", []))
            for row in rows:
                yield self._coerce(row)
        elif path.suffix == ".parquet":
            try:
                import pandas as pd  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("parquet support requires pandas/pyarrow") from exc
            for row in pd.read_parquet(path).to_dict("records"):
                yield self._coerce(row)

    def _coerce(self, row: Dict[str, Any]) -> DataItem:
        language = row.get("language") or row.get("extra_info", {}).get("language") or row.get("extra_infos", {}).get("language") or "python"
        source = row.get("source") or row.get("old_source") or row.get("original") or row.get("code") or ""
        patch = row.get("patch") or row.get("update") or row.get("diff") or ""
        reference = row.get("reference") or row.get("new_source") or row.get("ground_truth") or row.get("updated_code") or ""
        return DataItem(language=language, source=source, patch=patch, reference=reference, metadata=dict(row))
