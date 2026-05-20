from __future__ import annotations

"""AST/rule based data filter from the FastApply data-cleaning screenshots."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import typer

from rl_master.applications.fastapply.data.ast import AbstractSyntaxTree, build_ast
from rl_master.applications.fastapply.data.types import ChangeContext, Hunk, Unit
from rl_master.applications.fastapply.data.utils import expand_context_to_all_units
from .io import DatasetLike, as_rows, load, save
from .make import build as build_patch

app = typer.Typer(help="Filter data by AST scope and screenshot-reproduced rules.")

SUPPORTED_EXTENSIONS = {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp"}


def _split_hunks(diff: str) -> list[Hunk]:
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
    return [Hunk("\n".join(hunk)) for hunk in hunks]


def _all_units(units: list[Unit]) -> list[Unit]:
    out: list[Unit] = []

    def dfs(unit: Unit) -> None:
        out.append(unit)
        for child in unit.children:
            dfs(child)

    for unit in units:
        dfs(unit)
    return out


def _is_modified(unit: Unit, context: ChangeContext) -> bool:
    state = context.visit(unit)
    return bool(state and state.is_modified)


def is_only_function_and_class_modification(tree: AbstractSyntaxTree, context: ChangeContext, units: list[Unit]) -> bool:
    """Reject changes that touch comments/imports/docstrings without functional scope."""

    allowed = tree.get_unit_types("class & function") | tree.get_unit_types("statement")
    for unit in _all_units(units):
        if _is_modified(unit, context) and unit.type not in allowed:
            # Allow a class/function to own children; comments/imports are non-functional.
            return False
    return True


def has_import_modification(tree: AbstractSyntaxTree, context: ChangeContext, units: list[Unit]) -> bool:
    for unit in _all_units(units):
        if _is_modified(unit, context) and unit.type == "import":
            return True
    return False


def is_functional_modification(tree: AbstractSyntaxTree, context: ChangeContext, units: list[Unit]) -> bool:
    functional = tree.get_unit_types("functional") | tree.get_unit_types("class & function")
    for unit in _all_units(units):
        if _is_modified(unit, context) and unit.type in functional:
            return True
    return False


class Rule(ABC):
    def __init__(
        self,
        old_ast: AbstractSyntaxTree,
        old_context: ChangeContext,
        old_units: list[Unit],
        new_ast: AbstractSyntaxTree,
        new_context: ChangeContext,
        new_units: list[Unit],
    ):
        self._old_ast = old_ast
        self._old_context = old_context
        self._old_units = old_units
        self._new_ast = new_ast
        self._new_context = new_context
        self._new_units = new_units

    @abstractmethod
    def __call__(self) -> bool:
        pass


class EnsuresOnlyFunctionalAndClassModification(Rule):
    def __call__(self) -> bool:
        return is_only_function_and_class_modification(self._old_ast, self._old_context, self._old_units) and is_only_function_and_class_modification(self._new_ast, self._new_context, self._new_units)


class EnsuresHasImportModification(Rule):
    def __call__(self) -> bool:
        return has_import_modification(self._old_ast, self._old_context, self._old_units) or has_import_modification(self._new_ast, self._new_context, self._new_units)


class EnsuresHasFunctionalModification(Rule):
    def __call__(self) -> bool:
        return is_functional_modification(self._old_ast, self._old_context, self._old_units) or is_functional_modification(self._new_ast, self._new_context, self._new_units)


class Filter:
    _pattern = re.compile(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")

    def __init__(self, require_import: bool = True, only_functional_and_class: bool = False, max_changes: int = 7, max_chars: int = 131072):
        self.require_import = require_import
        self.only_functional_and_class = only_functional_and_class
        self.max_changes = max_changes
        self.max_chars = max_chars

    def __call__(self, language: str, old_source: str, new_source: str, diff: str) -> bool:
        old_ast = build_ast(language, old_source)
        new_ast = build_ast(language, new_source)
        hunks = _split_hunks(diff)
        if not hunks or len(hunks) > self.max_changes:
            return False
        if len(old_source) + len(new_source) + len(diff) > self.max_chars:
            return False
        old_units = old_ast.get_units_within_top_level_scope()
        new_units = new_ast.get_units_within_top_level_scope()
        old_context = ChangeContext("-", old_units, hunks)
        new_context = ChangeContext("+", new_units, hunks)
        expand_context_to_all_units(old_units, old_context)
        expand_context_to_all_units(new_units, new_context)

        rules: list[Rule] = [
            EnsuresHasFunctionalModification(old_ast, old_context, old_units, new_ast, new_context, new_units),
        ]
        if self.only_functional_and_class:
            rules.append(EnsuresOnlyFunctionalAndClassModification(old_ast, old_context, old_units, new_ast, new_context, new_units))
        if self.require_import:
            rules.append(EnsuresHasImportModification(old_ast, old_context, old_units, new_ast, new_context, new_units))
        return all(rule() for rule in rules)


def filter_row(row: Dict[str, Any], require_import: bool = True) -> bool:
    filename = str(row.get("file", ""))
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        return False
    language = str(row.get("language") or "python")
    if language not in {"python", "java", "javascript", "typescript", "go", "rust", "c", "cpp"}:
        return False
    try:
        return Filter(require_import=require_import)(language, str(row.get("old_source", "")), str(row.get("new_source", "")), str(row.get("diff", "")))
    except (AbstractSyntaxTree.EmptySourceError, ValueError, SyntaxError):
        return False


def process(dataset: Any, require_import: bool = True) -> DatasetLike:
    return [row for row in as_rows(dataset) if filter_row(row, require_import=require_import)]


@app.command(name="filter")
def main(
    dataset_load_path: Path = typer.Option(..., help="dataset load path"),
    dataset_save_path: Optional[Path] = typer.Option(None, help="dataset save path"),
    require_import: bool = typer.Option(True, help="mirror screenshot rule requiring import modification"),
):
    path_to_save = dataset_save_path or dataset_load_path
    rows = load(dataset_load_path)
    out = process(rows, require_import=require_import)
    save(path_to_save, out)
    typer.echo(f"filtered {len(rows)} -> {len(out)} rows at {path_to_save}")


__all__ = [
    "Filter",
    "Rule",
    "EnsuresOnlyFunctionalAndClassModification",
    "EnsuresHasImportModification",
    "EnsuresHasFunctionalModification",
    "is_only_function_and_class_modification",
    "has_import_modification",
    "is_functional_modification",
    "filter_row",
    "process",
    "app",
]
