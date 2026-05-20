from __future__ import annotations

import ast
from typing import Optional

from .ast_base import AbstractSyntaxTree
from .ast_utils import register
from ..types import Span, Unit


def _node_span(node: ast.AST) -> Optional[Span]:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None) or lineno
    if lineno is None:
        return None
    return Span(int(lineno), int(end_lineno))


@register("python")
class AbstractSyntaxTreePython(AbstractSyntaxTree):
    def __init__(self, source: str):
        self._py_ast: Optional[ast.Module] = None
        super().__init__("python", source)

    def build(self) -> None:
        try:
            self._py_ast = ast.parse(self._source)
        except SyntaxError as exc:
            raise ValueError(str(exc)) from exc
        self._build_module(self._py_ast, self._root)

    def _build_module(self, node: ast.Module, parent: Unit) -> None:
        for child in node.body:
            unit = self._convert(child, parent)
            if unit is not None:
                parent.add_child(unit)

    def _convert(self, node: ast.AST, parent: Unit) -> Optional[Unit]:
        span = _node_span(node)
        if span is None:
            return None
        data = {"span": span, "header": self.get_source_code(span).splitlines()[0] if self.get_source_code(span).splitlines() else ""}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            data["type"] = "import"
            return Unit(node, data, parent)
        if isinstance(node, ast.ClassDef):
            data.update({"type": "class", "name": node.name, "is_decorated": bool(node.decorator_list)})
            unit = Unit(node, data, parent)
            self._add_docstring(node, unit)
            for body_node in node.body:
                child = self._convert(body_node, unit)
                if child is not None:
                    unit.add_child(child)
            return unit
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            data.update({"type": "function", "name": node.name, "is_decorated": bool(node.decorator_list), "is_async": isinstance(node, ast.AsyncFunctionDef)})
            unit = Unit(node, data, parent)
            self._add_docstring(node, unit)
            for body_node in node.body:
                if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nested = self._convert(body_node, unit)
                    if nested is not None:
                        unit.add_child(nested)
            return unit
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            data["type"] = "docstring"
            return Unit(node, data, parent)
        if isinstance(node, ast.stmt):
            data["type"] = "statement"
            return Unit(node, data, parent)
        return None

    def _add_docstring(self, node: ast.AST, unit: Unit) -> None:
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            span = _node_span(body[0])
            if span is not None:
                unit.add_child(Unit(body[0], {"type": "docstring", "span": span, "header": self.get_source_code(span).strip()}, unit))


__all__ = ["AbstractSyntaxTreePython"]
