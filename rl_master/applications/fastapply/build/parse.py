from __future__ import annotations

"""Parse raw git diffs into SEARCH/REPLACE change blocks.

This mirrors the screenshot file ``RL-master/applications/fastapply/build/parse.py``:
raw rows with ``old_source``, ``new_source`` and unified ``diff`` are converted
into a structured ``changes`` list where every hunk has an old SEARCH body and a
new REPLACE body.
"""

import random
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import typer

from rl_master.applications.fastapply.data.types import Hunk
from .io import DatasetLike, as_rows, concatenate_datasets, dataset_from_list, load, save

app = typer.Typer(help="Parse structured change content from a git diff.")


class Parser:
    _pattern = re.compile(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")

    def __call__(self, language: str, old_source: str, new_source: str, diff: str) -> Optional[list[dict[str, str]]]:
        hunks = self.split_hunks(diff)
        if not hunks:
            return None
        return [
            {
                "old": self.build(Hunk(hunk), version="old"),
                "new": self.build(Hunk(hunk), version="new"),
            }
            for hunk in hunks
        ]

    @staticmethod
    def split_hunks(diff: str) -> list[str]:
        hunks: list[list[str]] = []
        current: list[str] = []
        for line in diff.splitlines():
            if line.startswith("@@"):
                if current:
                    hunks.append(current)
                current = [line]
            elif current:
                current.append(line)
        if current:
            hunks.append(current)
        return ["\n".join(hunk) for hunk in hunks]

    @staticmethod
    def build(hunk: Hunk, version: Literal["old", "new"]) -> str:
        marker = "-" if version == "old" else "+"
        opposite = "+" if version == "old" else "-"
        result: list[str] = []
        for line in hunk.body.splitlines():
            if line.startswith("\\"):
                continue
            if line.startswith(opposite):
                continue
            if line.startswith(marker) or line.startswith(" "):
                result.append(line[1:])
            else:
                result.append(line)
        return "\n".join(result)


def parse_row(item: Dict[str, Any]) -> Dict[str, Any]:
    parser = Parser()
    item = dict(item)
    changes = parser(item.get("language", ""), item.get("old_source", ""), item.get("new_source", ""), item.get("diff", ""))
    item["changes"] = changes or []
    return item


def process(dataset: Any) -> DatasetLike:
    return [parse_row(row) for row in as_rows(dataset)]


def split(dataset: DatasetLike, split: float = 0.95, seed: int = 42) -> tuple[DatasetLike, DatasetLike]:
    """Split while trying to keep all samples from the same file together."""

    random.seed(seed)
    file_to_samples: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset):
        file_to_samples.setdefault(str(sample.get("file", sample.get("id", index))), []).append(index)
    unique_files = list(file_to_samples)
    random.shuffle(unique_files)
    train_indices: list[int] = []
    test_indices: list[int] = []
    target_train = int(len(dataset) * split)
    for file in unique_files:
        indices = file_to_samples[file]
        if len(train_indices) + len(indices) <= target_train:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)
    if not test_indices and train_indices:
        test_indices.append(train_indices.pop())
    return [dataset[i] for i in train_indices], [dataset[i] for i in test_indices]


def split_dataset(datasets: dict[str, DatasetLike], p: float = 0.9) -> tuple[DatasetLike, DatasetLike]:
    train_datasets: list[DatasetLike] = []
    test_datasets: list[DatasetLike] = []
    for name, dataset in datasets.items():
        train_dataset, test_dataset = split(dataset, p)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)


@app.command(name="parse")
def main(
    dataset_load_path: Path = typer.Option(..., help="dataset load path (json/jsonl/parquet)"),
    dataset_save_path: Path = typer.Option(..., help="dataset save path"),
    workers: int = typer.Option(8, help="compatibility option; local list processing is synchronous"),
):
    rows = process(load(dataset_load_path))
    save(dataset_save_path, rows)
    typer.echo(f"parsed {len(rows)} rows -> {dataset_save_path}")


__all__ = ["Parser", "parse_row", "process", "split", "split_dataset", "app"]
