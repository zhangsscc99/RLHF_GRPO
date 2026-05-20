from __future__ import annotations

"""Repository helpers reproduced from the FastApply build screenshots."""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional


class Repo:
    def __init__(
        self,
        dir: str | Path,
        url: Optional[str] = None,
        name: Optional[str] = None,
        branch: str = "main",
        lang: Optional[str] = None,
    ):
        self._dir = Path(dir)
        self._url = url
        self._name = name or self._infer_name()
        self._branch = branch
        self._lang = lang

    def _infer_name(self) -> str:
        if self._url:
            match = re.search(r"/([^/]+?)(?:\.git)?$", self._url)
            if match:
                return match.group(1)
        return os.path.basename(str(self._dir)) or "repo"

    @property
    def dir(self) -> str:
        return str(self._dir)

    @property
    def url(self) -> Optional[str]:
        return self._url

    @property
    def name(self) -> str:
        return self._name

    @property
    def branch(self) -> str:
        return self._branch

    @property
    def lang(self) -> Optional[str]:
        return self._lang

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"dir": self.dir, "url": self.url, "branch": self.branch, "name": self.name, "lang": self.lang}


def _run(cmd: list[str], cwd: str | Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def clone_repo(url: str, dir: str | Path, proxy: Optional[str] = None) -> Path:
    """Clone a repository or reuse an existing local checkout.

    GitPython is optional in the screenshot code; this adaptation uses the git CLI
    so the pipeline remains runnable in the minimal training environment.
    """

    target = Path(dir)
    if (target / ".git").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if proxy or env.get("PROXY"):
        env["http_proxy"] = proxy or env.get("PROXY", "")
        env["https_proxy"] = proxy or env.get("PROXY", "")
    subprocess.run(["git", "clone", url, str(target)], check=True, env=env)
    return target


__all__ = ["Repo", "clone_repo"]
