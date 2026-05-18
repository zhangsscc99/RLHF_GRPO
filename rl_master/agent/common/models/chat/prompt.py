from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    @dataclass(frozen=True)
    class Variable:
        name: str
        path: str = ""
        metadata: Optional[Dict[str, Any]] = None
        description: Optional[str] = None

    content: str
    variables: List[Variable] = field(default_factory=list)
    sys_content: Optional[str] = None
    sys_variables: List[Variable] = field(default_factory=list)

    def render(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        def get(path: str, default: Any = "") -> Any:
            cur: Any = data
            if not path:
                return cur
            for part in path.split("."):
                if isinstance(cur, dict):
                    cur = cur.get(part, default)
                else:
                    return default
            return cur

        def fill(text: str, variables: List[PromptTemplate.Variable]) -> str:
            out = text
            for var in variables:
                value = get(var.path or var.name, "")
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                out = re.sub(r"{{\s*" + re.escape(var.name) + r"\s*}}", value, out)
            return out

        messages: List[Dict[str, str]] = []
        if self.sys_content is not None:
            messages.append({"role": "system", "content": fill(self.sys_content, self.sys_variables)})
        messages.append({"role": "user", "content": fill(self.content, self.variables)})
        return messages

    @classmethod
    def from_file(cls, path: Path | str) -> "PromptTemplate":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        sys_content = None
        user_content = None
        for item in data:
            if "system" in item:
                sys_content = item["system"]
            if "user" in item:
                user_content = item["user"]
        if user_content is None:
            user_content = data.get("user") if isinstance(data, dict) else ""
        vars_ = [cls.Variable(name=m) for m in sorted(set(re.findall(r"{{\s*([\w.]+)\s*}}", user_content or "")))]
        sys_vars = [cls.Variable(name=m) for m in sorted(set(re.findall(r"{{\s*([\w.]+)\s*}}", sys_content or "")))]
        return cls(content=user_content or "", variables=vars_, sys_content=sys_content, sys_variables=sys_vars)
