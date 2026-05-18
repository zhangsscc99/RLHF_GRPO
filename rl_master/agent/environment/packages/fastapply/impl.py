from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rl_master.agent.common.models import PromptTemplate
from rl_master.agent.environment.manager import DataItem, GlobalDatasetManager
from rl_master.agent.evaluation.extractors import extract_tagged_content
from rl_master.applications.fastapply.reward.utils import build, compare

TERMINAL_STATE = "TERMINAL"
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
        self._path = path
        self._seed = seed
        self._mode = "train" if is_train else "val"
        self._template = PromptTemplate(content=TEMPLATE, variables=[PromptTemplate.Variable("language"), PromptTemplate.Variable("source"), PromptTemplate.Variable("patch")], sys_content=SYSTEM_PROMPT)
        self._dataset_manager = GlobalDatasetManager(path, split=self._mode) if path else None
        self._state: Optional[FastApplyState] = None

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._seed = seed
        item = self._dataset_manager.get_data_item(self._seed) if self._dataset_manager else None
        if item is None:
            item = DataItem(
                language="python",
                source="def add(a, b):\n    return a + b\n",
                patch=build([{"old": "def add(a, b):\n    return a + b\n", "new": "def add(a, b):\n    return a + b + 1\n"}]),
                reference="def add(a, b):\n    return a + b + 1\n",
                metadata={"fixture": "tiny-add"},
            )
        self._state = FastApplyState(language=item.language, source=item.source, patch=item.patch, reference=item.reference)
        return {"language": self._state.language, "source": self._state.source, "patch": self._state.patch}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()
        assert self._state is not None
        tagged = extract_tagged_content(action, tag="updated")
        candidate = tagged if tagged is not None else action
        is_valid = tagged is not None and bool(candidate.strip())
        correctness = 1.0 if compare(self._state.reference, candidate) else 0.0
        reward = correctness * 0.8 + (1.0 if is_valid else 0.0) * 0.2
        info = {
            "metrics": {"action_is_valid": is_valid, "success": correctness > 0.9, "correctness": correctness},
            "metrics_agg_mode": {"action_is_valid": "last", "success": "last", "correctness": "last"},
        }
        return TERMINAL_STATE, reward, True, False, info

    def get(self, name: str) -> PromptTemplate:
        if name != "template":
            raise ValueError(f"{name} is not supported")
        return self._template
