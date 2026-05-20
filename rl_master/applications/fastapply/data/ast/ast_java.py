from __future__ import annotations

import re
from typing import Optional

from .ast_base import AbstractSyntaxTree
from .ast_utils import register
from ..types import Span, Unit


@register("java")
class AbstractSyntaxTreeJava(AbstractSyntaxTree):
    def __init__(self, source: str):
        super().__init__("java", source)

    def build(self) -> None:
        lines = self._source.splitlines()
        consumed: set[int] = set()
        class_units: list[Unit] = []

        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("package "):
                self._root.add_child(Unit(None, {"type": "package", "span": Span(index, index), "header": stripped}, self._root))
                consumed.add(index)
            elif stripped.startswith("import "):
                self._root.add_child(Unit(None, {"type": "import", "span": Span(index, index), "header": stripped}, self._root))
                consumed.add(index)

        for index, line in enumerate(lines, start=1):
            if index in consumed:
                continue
            match = re.search(r"\b(class|interface|enum|record)\s+([A-Za-z_]\w*)", line)
            if not match:
                continue
            end = self._find_block_end(index)
            unit = Unit(None, {"type": "class", "name": match.group(2), "span": Span(index, end), "header": line.strip(), "is_decorated": "@" in lines[max(0, index - 3): index - 1]}, self._root)
            self._root.add_child(unit)
            class_units.append(unit)
            consumed.update(range(index, end + 1))
            self._add_methods(unit)

        for index, line in enumerate(lines, start=1):
            if index in consumed or not line.strip():
                continue
            if line.strip().startswith(("//", "/*", "*")):
                typ = "comment"
            else:
                typ = "statement"
            self._root.add_child(Unit(None, {"type": typ, "span": Span(index, index), "header": line.strip()}, self._root))

    def _find_block_end(self, start_line: int) -> int:
        lines = self._source.splitlines()
        balance = 0
        seen = False
        for index in range(start_line, len(lines) + 1):
            text = lines[index - 1]
            balance += text.count("{") - text.count("}")
            seen = seen or "{" in text
            if seen and balance <= 0:
                return index
        return len(lines) or start_line

    def _add_methods(self, class_unit: Unit) -> None:
        lines = self._source.splitlines()
        start, end = class_unit.span.start, class_unit.span.end
        method_pattern = re.compile(r"(?:public|private|protected|static|final|synchronized|abstract|native|\s)+[\w<>\[\], ?]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws [^{]+)?\{")
        control = {"if", "for", "while", "switch", "catch", "try", "return", "new"}
        index = start + 1
        while index < end:
            line = lines[index - 1]
            match = method_pattern.search(line)
            if match and match.group(1) not in control:
                method_end = self._find_block_end(index)
                class_unit.add_child(Unit(None, {"type": "function", "name": match.group(1), "span": Span(index, min(method_end, end)), "header": line.strip()}, class_unit))
                index = method_end + 1
            else:
                index += 1


__all__ = ["AbstractSyntaxTreeJava"]
