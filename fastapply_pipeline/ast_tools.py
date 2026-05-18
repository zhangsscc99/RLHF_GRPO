from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScopeInfo:
    language: str
    parse_ok: bool
    top_level_symbols: list[str]
    error: str = ""
    checker: str = "heuristic"
    checked: bool = True


def analyze_python(source: str) -> ScopeInfo:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ScopeInfo("python", False, [], str(exc), checker="python.ast", checked=True)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.append("import")
    return ScopeInfo("python", True, names, checker="python.ast", checked=True)


LANG_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "java": "java",
    "golang": "go",
    "rs": "rust",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "rb": "ruby",
    "php": "php",
    "lua": "lua",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "html": "html",
    "css": "css",
}

EXTENSIONS = {
    "javascript": ".js",
    "typescript": ".ts",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "c": ".c",
    "cpp": ".cpp",
    "csharp": ".cs",
    "shell": ".sh",
    "ruby": ".rb",
    "php": ".php",
    "lua": ".lua",
}


def _normalize_language(language: str) -> str:
    key = (language or "").strip().lower()
    return LANG_ALIASES.get(key, key or "text")


def _symbols_by_regex(language: str, source: str) -> list[str]:
    patterns = [
        r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
        r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=",
        r"(?:class|interface|enum)\s+([A-Za-z_$][\w$]*)",
        r"\bfunc\s+([A-Za-z_]\w*)\s*\(",
        r"\bfn\s+([A-Za-z_]\w*)\s*\(",
        r"\b(?:def|class)\s+([A-Za-z_]\w*)",
        r"\b(?:public\s+)?(?:class|interface|enum)\s+([A-Za-z_]\w*)",
    ]
    names: list[str] = []
    for pat in patterns:
        names.extend(re.findall(pat, source))
    if language in {"c", "cpp"}:
        names.extend(re.findall(r"^[A-Za-z_][\w\s\*:&<>]*\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", source, re.MULTILINE))
    return list(dict.fromkeys(names))


def _run_checker(command: list[str], source: str, suffix: str, filename: str | None = None, timeout: float = 3.0) -> tuple[bool, str, str]:
    executable = command[0]
    if shutil.which(executable) is None:
        return True, "", f"{executable} not installed; syntax check skipped"
    with tempfile.TemporaryDirectory(prefix="fastapply_syntax_") as tmpdir:
        path = Path(tmpdir) / (filename or f"snippet{suffix}")
        path.write_text(source, encoding="utf-8")
        cmd = [part.format(file=str(path), dir=tmpdir) for part in command]
        try:
            proc = subprocess.run(cmd, cwd=tmpdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, executable, "syntax checker timed out"
        output = (proc.stderr or proc.stdout or "").strip()
        return proc.returncode == 0, executable, output


def _java_filename(source: str) -> str:
    m = re.search(r"\bpublic\s+(?:class|interface|enum)\s+([A-Za-z_]\w*)", source)
    return f"{m.group(1)}.java" if m else "Snippet.java"


def _check_with_tool(language: str, source: str) -> tuple[bool, str, str, bool]:
    if language == "javascript":
        ok, checker, error = _run_checker(["node", "--check", "{file}"], source, ".js")
        return ok, checker, error, "not installed" not in error
    if language == "typescript":
        # `tsc --noEmit` is the best local no-run checker if TypeScript is installed.
        ok, checker, error = _run_checker(["tsc", "--noEmit", "--skipLibCheck", "{file}"], source, ".ts", timeout=5.0)
        return ok, checker, error, "not installed" not in error
    if language == "java":
        ok, checker, error = _run_checker(["javac", "-Xlint:none", "{file}"], source, ".java", filename=_java_filename(source), timeout=5.0)
        return ok, checker, error, "not installed" not in error
    if language == "go":
        ok, checker, error = _run_checker(["gofmt", "-e", "{file}"], source, ".go")
        return ok, checker, error, "not installed" not in error
    if language == "rust":
        ok, checker, error = _run_checker(["rustc", "--crate-type=lib", "--emit=metadata", "{file}", "-o", "{dir}/out.rmeta"], source, ".rs", timeout=5.0)
        return ok, checker, error, "not installed" not in error
    if language == "c":
        ok, checker, error = _run_checker(["gcc", "-fsyntax-only", "{file}"], source, ".c", timeout=5.0)
        return ok, checker, error, "not installed" not in error
    if language == "cpp":
        ok, checker, error = _run_checker(["g++", "-fsyntax-only", "{file}"], source, ".cpp", timeout=5.0)
        return ok, checker, error, "not installed" not in error
    if language == "shell":
        ok, checker, error = _run_checker(["bash", "-n", "{file}"], source, ".sh")
        return ok, checker, error, "not installed" not in error
    if language == "ruby":
        ok, checker, error = _run_checker(["ruby", "-c", "{file}"], source, ".rb")
        return ok, checker, error, "not installed" not in error
    if language == "php":
        ok, checker, error = _run_checker(["php", "-l", "{file}"], source, ".php")
        return ok, checker, error, "not installed" not in error
    if language == "lua":
        ok, checker, error = _run_checker(["luac", "-p", "{file}"], source, ".lua")
        return ok, checker, error, "not installed" not in error
    return True, "heuristic", "", False


def analyze_by_external_or_heuristic(language: str, source: str) -> ScopeInfo:
    names = _symbols_by_regex(language, source)
    if language == "json":
        try:
            json.loads(source)
            return ScopeInfo(language, True, names, checker="json.loads", checked=True)
        except Exception as exc:
            return ScopeInfo(language, False, names, str(exc), checker="json.loads", checked=True)
    if language == "yaml":
        try:
            import yaml  # type: ignore

            yaml.safe_load(source)
            return ScopeInfo(language, True, names, checker="yaml.safe_load", checked=True)
        except Exception as exc:
            return ScopeInfo(language, False, names, str(exc), checker="yaml.safe_load", checked=True)

    ok, checker, error, checked = _check_with_tool(language, source)
    if checked:
        return ScopeInfo(language, ok, names, "" if ok else error, checker=checker, checked=True)

    # Fallback: common text languages such as TS/C#/HTML/CSS may lack a local
    # parser on this server.  Keep the reward non-blocking but mark checked=False.
    basic_ok = bool(source.strip())
    return ScopeInfo(language, basic_ok, names, "" if basic_ok else "empty source", checker="heuristic", checked=False)


def analyze_source(language: str, source: str) -> ScopeInfo:
    language = _normalize_language(language)
    if language == "python":
        return analyze_python(source)
    return analyze_by_external_or_heuristic(language, source)
