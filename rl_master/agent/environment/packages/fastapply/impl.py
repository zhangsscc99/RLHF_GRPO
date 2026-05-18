from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rl_master.agent.common.models import GenerationConfig, Inference, LLM, PromptTemplate, RespondWithPattern
from rl_master.agent.evaluation.extractors import extract_tagged_content
from rl_master.applications.fastapply.reward.utils import build, compare

SYSTEM_PROMPT = "You are a coding assistant specialized in merging code updates, ensuring all changes are fully integrated."
TEMPLATE = """Merge all changes from the <patch> snippet into the source code below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated source code, enclosed within <updated> and </updated> tags.
- Do not include explanations or code fences.

```{{ language }}
{{ source }}
```

<patch>
{{ patch }}
</patch>
"""


@dataclass
class FastApplyState:
    language: str
    source: str
    patch: str
    reference: str


class FastApply:
    def __init__(self, path: str = "", seed: int = 0, is_train: bool = True):
        self._seed = seed
        self._mode = "train" if is_train else "val"
        self._template = PromptTemplate(content=TEMPLATE, variables=[PromptTemplate.Variable("language"), PromptTemplate.Variable("source"), PromptTemplate.Variable("patch")], sys_content=SYSTEM_PROMPT)
        self._state: Optional[FastApplyState] = None

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._seed = seed
        self._state = FastApplyState(language="python", source="def add(a, b):\n    return a + b\n", patch=build([{"old": "def add(a, b):\n    return a + b\n", "new": "def add(a, b):\n    return a + b + 1\n"}]), reference="def add(a, b):\n    return a + b + 1\n")
        return {"language": self._state.language, "source": self._state.source, "patch": self._state.patch}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()
        assert self._state is not None
        candidate = extract_tagged_content(action, tag="updated") or action
        is_valid = bool(candidate)
        correctness = 1.0 if compare(self._state.reference, candidate) else 0.0
        reward = correctness * 0.8 + (1.0 if is_valid else 0.0) * 0.2
        info = {"metrics": {"action_is_valid": is_valid, "success": correctness > 0.9}, "metrics_agg_mode": {"action_is_valid": "last", "success": "last"}}
        return "TERMINAL", reward, True, False, info

    def get(self, name: str) -> PromptTemplate:
        if name != "template":
            raise ValueError(f"{name} is not supported")
        return self._template
