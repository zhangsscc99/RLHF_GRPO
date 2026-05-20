from __future__ import annotations

"""Extract raw old/new/diff rows from git repositories.

Reproduces the screenshot stage ``build/extract.py`` with a dependency-light git
CLI backend.  It walks single-parent commits, keeps source-code files, and emits
rows consumed by ``parse.py``.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional
from uuid import uuid4

import typer

from rl_master.applications.fastapply.data.ast import detect_language
from rl_master.applications.fastapply.data.ops import Repo, clone_repo
from .io import DatasetLike, save

app = typer.Typer(help="Extract raw data from a git repository.")

SOURCE_SUFFIXES = {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs"}


def _git(repo_dir: str | Path, args: list[str], check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout


class Extractor:
    """Code snapshot extractor."""

    def __init__(self, repo: Repo):
        self._repo = repo
        self._repo_dir = Path(repo.dir)
        if repo.url and not (self._repo_dir / ".git").exists():
            clone_repo(repo.url, self._repo_dir)
        self._repo_branch = repo.branch

    def __call__(self, branch: Literal["master", "main", "dev"] | str = "main") -> Iterable[Dict[str, Any]]:
        commits = self._get_commits(branch)
        for commit in commits:
            for diff in self._get_commit_diffs(commit):
                try:
                    language = detect_language(diff["path"])
                except ValueError:
                    continue
                old_source = self._get_blob_source(f"{commit}^", diff["path"])
                new_source = self._get_blob_source(commit, diff["path"])
                if not old_source.strip() or not new_source.strip():
                    continue
                yield {
                    "id": uuid4().hex,
                    "repo": self._repo.name,
                    "file": diff["path"],
                    "commit": commit,
                    "language": language,
                    "old_source": old_source,
                    "new_source": new_source,
                    "diff": diff["diff"],
                }

    def run(self) -> DatasetLike:
        return list(self(self._repo_branch))

    def _get_commits(self, branch: str) -> list[str]:
        out = _git(self._repo_dir, ["rev-list", "--first-parent", branch])
        commits = [line.strip() for line in out.splitlines() if line.strip()]
        # Keep single-parent commits only, as shown in the screenshots.
        return [commit for commit in commits if len(_git(self._repo_dir, ["show", "-s", "--format=%P", commit]).split()) == 1]

    def _get_commit_diffs(self, commit: str) -> list[Dict[str, str]]:
        name_status = _git(self._repo_dir, ["diff-tree", "--no-commit-id", "--name-only", "-r", commit])
        rows: list[Dict[str, str]] = []
        for path in [line.strip() for line in name_status.splitlines() if line.strip()]:
            suffix = Path(path).suffix.lower()
            if suffix not in SOURCE_SUFFIXES:
                continue
            patch = _git(self._repo_dir, ["diff", f"{commit}^", commit, "--", path], check=False)
            if "@@" not in patch:
                continue
            rows.append({"path": path, "diff": patch})
        return rows

    def _get_blob_source(self, rev: str, path: str) -> str:
        return _git(self._repo_dir, ["show", f"{rev}:{path}"], check=False)


@app.command(name="extract")
def main(
    repo: Optional[str] = typer.Option(None, help="repo config JSON or URL"),
    path: Optional[Path] = typer.Option(None, help="output path"),
):
    if repo is None:
        raise typer.BadParameter("Please provide --repo")
    config = json.loads(repo) if repo.strip().startswith("{") else {"url": repo, "dir": Path(repo).stem, "branch": "main"}
    repo_obj = Repo(**config)
    rows = Extractor(repo_obj).run()
    output = path or Path(f"{repo_obj.name}.jsonl")
    save(output, rows)
    typer.echo(f"dataset {repo_obj.name} has been saved to {output}")


__all__ = ["Extractor", "app"]
