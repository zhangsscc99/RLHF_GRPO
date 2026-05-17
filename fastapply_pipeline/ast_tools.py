from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeInfo:
    language: str
    parse_ok: bool
    top_level_symbols: list[str]
    error: str = ""


def analyze_python(source: str) -> ScopeInfo:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ScopeInfo("python", False, [], str(exc))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.append("import")
    return ScopeInfo("python", True, names)


def analyze_javascript(source: str) -> ScopeInfo:
    # Lightweight regex fallback; the photographed pipeline mentions AST boundaries.
    # Real deployment should replace this with tree-sitter.
    names = re.findall(r"(?:export\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", source)
    names += re.findall(r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", source)
    return ScopeInfo("javascript", True, names)


def analyze_source(language: str, source: str) -> ScopeInfo:
    if language == "python":
        return analyze_python(source)
    if language in {"javascript", "typescript"}:
        return analyze_javascript(source)
    return ScopeInfo(language, True, [])
