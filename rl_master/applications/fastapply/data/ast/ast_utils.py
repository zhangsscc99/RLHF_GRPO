from __future__ import annotations

"""Parser registry and language detection for the FastApply data-cleaning AST layer."""

from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Type

_AST: Dict[str, type] = {}


def register(language: Literal["python", "java"] | str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        _AST[str(language).lower()] = cls
        return cls

    return decorator


def detect_language(filename: str, code: Optional[str] = None) -> str:
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".sh": "shell",
    }
    if suffix in mapping:
        return mapping[suffix]
    if code:
        stripped = code.lstrip()
        if stripped.startswith("package ") or " public class " in f" {code[:500]} ":
            return "java"
        if stripped.startswith("def ") or stripped.startswith("import ") or "\nclass " in code[:500]:
            return "python"
    raise ValueError("unsupported language")


def get_ast_parser(name: Literal["python", "java"] | str) -> type:
    key = str(name).lower()
    if key not in _AST:
        # Lazy imports keep module import cheap and avoid circular imports.
        from . import ast_python as _py  # noqa: F401
        from . import ast_java as _java  # noqa: F401
        from .ast_base import AbstractSyntaxTree

        if key not in _AST:
            _AST[key] = AbstractSyntaxTree
    return _AST[key]


def build_ast(language: Literal["python", "java"] | str, source: str):
    cls = get_ast_parser(language)
    return cls(source) if cls.__name__.endswith(("Python", "Java")) else cls(language, source)


def debug_ast(root, indent: int = 0, only_print_type: bool = False) -> None:
    def dfs(node, depth: int) -> None:
        if node is None:
            return
        prefix = " " * depth
        if only_print_type:
            print(f"{prefix}{node.type}")
        else:
            print(f"{prefix}{node.type} {node.span} {node.data}")
        for child in node.children:
            dfs(child, depth + 2)

    dfs(root, indent)


__all__ = ["register", "detect_language", "get_ast_parser", "build_ast", "debug_ast"]
