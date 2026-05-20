from __future__ import annotations

"""Abstract/lightweight syntax tree used by data-cleaning filters.

The screenshots use tree-sitter.  This repo keeps a compatible Unit-based API and
uses Python's stdlib AST plus regex/brace heuristics as a runtime fallback so the
pipeline runs without installing native tree-sitter grammars.
"""

import re
from abc import ABC
from typing import Literal, Optional, Sequence, Set, Union

from ..types import Span, Unit


class AbstractSyntaxTree(ABC):
    class EmptySourceError(Exception):
        pass

    def __init__(self, language: Literal["python", "java"] | str, source: str):
        self._language = str(language).lower()
        self._source = source
        self._lines = source.splitlines(keepends=True)
        if not source.strip():
            raise AbstractSyntaxTree.EmptySourceError("please provide a non-empty source code file")
        end_line = max(1, len(source.splitlines()))
        self._root = Unit(None, {"type": "root", "name": self._language, "span": Span(1, end_line)})
        self.build()

    @property
    def tree(self):  # tree-sitter compatibility hook
        return None

    @property
    def root(self) -> Unit:
        return self._root

    @property
    def language(self) -> str:
        return self._language

    @property
    def source(self) -> str:
        return self._source

    def get_source_code(self, span: Union[Span, Unit, None] = None) -> str:
        if span is None:
            return self._source
        if isinstance(span, Unit):
            span = span.span
        return "".join(self._lines[span.start - 1 : span.end])

    def get_units_within_top_level_scope(self) -> list[Unit]:
        return self._root.children or [self._root]

    @staticmethod
    def get_unit_types(tag: Literal[
        "special",
        "comment",
        "docstring",
        "comment & docstring",
        "class",
        "function",
        "class & function",
        "statement",
        "non-functional",
        "functional",
    ] | str) -> Set[str]:
        match tag:
            case "special":
                return {"ignored", "package", "import"}
            case "comment":
                return {"comment"}
            case "docstring":
                return {"docstring"}
            case "comment & docstring":
                return {"comment", "docstring"}
            case "class":
                return {"class"}
            case "function":
                return {"function"}
            case "class & function":
                return {"class", "function"}
            case "statement":
                return {"statement"}
            case "non-functional":
                return {"comment", "docstring", "import", "package"}
            case "functional":
                return {"class", "function", "statement"}
            case _:
                return {str(tag)}

    def build(self) -> None:
        self._build_generic()

    def _build_generic(self) -> None:
        """Fallback parser for JS/TS/Go/etc.: imports + classes/functions + statements."""

        lines = self._source.splitlines()
        consumed: set[int] = set()
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("import ", "from ", "#include")):
                self._root.add_child(Unit(None, {"type": "import", "span": Span(index, index), "header": stripped}))
                consumed.add(index)
                continue
            if re.search(r"\b(class|interface|enum|struct)\s+[A-Za-z_]\w*", stripped):
                end = self._find_block_end(index)
                self._root.add_child(Unit(None, {"type": "class", "span": Span(index, end), "header": stripped}))
                consumed.update(range(index, end + 1))
                continue
            if re.search(r"\b(function|def|func|fn)\s+[A-Za-z_$]\w*\s*\(", stripped) or re.match(r"(?:export\s+)?(?:const|let|var)\s+[A-Za-z_$]\w*\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", stripped):
                end = self._find_block_end(index)
                self._root.add_child(Unit(None, {"type": "function", "span": Span(index, end), "header": stripped}))
                consumed.update(range(index, end + 1))
        for index, line in enumerate(lines, start=1):
            if index in consumed or not line.strip():
                continue
            self._root.add_child(Unit(None, {"type": "statement", "span": Span(index, index), "header": line.strip()}))

    def _find_block_end(self, start_line: int) -> int:
        lines = self._source.splitlines()
        balance = 0
        seen_brace = False
        for index in range(start_line, len(lines) + 1):
            line = lines[index - 1]
            balance += line.count("{") - line.count("}")
            seen_brace = seen_brace or "{" in line
            if seen_brace and balance <= 0:
                return index
            if not seen_brace and index > start_line and lines[index - 1].strip() and not lines[index - 1].startswith((" ", "\t")):
                return max(start_line, index - 1)
        return len(lines) or start_line

    def print_tree(self, node: Optional[Unit] = None, indent: int = 0) -> None:
        node = node or self._root
        print(" " * indent + f"{node.type} {node.span}")
        for child in node.children:
            self.print_tree(child, indent + 2)


__all__ = ["AbstractSyntaxTree"]
