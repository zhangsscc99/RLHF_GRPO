#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import numpy as np

try:
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover - keep the migrated evaluator runnable offline
    class Dataset(list):
        @classmethod
        def from_list(cls, rows):
            return cls(rows)

from rl_master.agent.common.models import GenerationConfig, LLM, PromptTemplate, RespondWithPattern
from rl_master.agent.evaluation.agents import SimpleQA
from rl_master.agent.evaluation.extractors import extract_tagged_content
from rl_master.agent.evaluation.pipeline import InferencePipeline


def get_template():
    sys_prompt = "You are a coding assistant that helps merge code updates, ensuring all changes are correctly integrated."
    prompt = """Merge all changes from the <patch> snippet into the <original-code> below.
<original-code>
{{ original }}
</original-code>
<patch>
{{ patch }}
</patch>
Provide the complete updated code."""
    return PromptTemplate(prompt, [PromptTemplate.Variable("original"), PromptTemplate.Variable("patch")], sys_content=sys_prompt)


async def main():
    dataset = Dataset.from_list([{"original": "def add(a,b):\n    return a+b\n", "patch": "<<<<<<< SEARCH\ndef add(a,b):\n    return a+b\n=======\ndef add(a,b):\n    return a+b+1\n>>>>>>> REPLACE", "reference": "def add(a,b):\n    return a+b+1\n"}])
    agent = SimpleQA(LLM("Qwen3-30B-A3B-Instruct"), get_template(), parse=RespondWithPattern(r"<updated>(.*?)</updated>"), config=GenerationConfig(temperature=0.2))
    pipe = InferencePipeline(agent)
    out = await pipe(dataset)
    scores = []
    for item in out:
        scores.append(1.0 if (item.get("result") or "").strip() == item["reference"].strip() else 0.0)
    print({"accuracy": float(np.mean(scores)), "rows": len(scores)})


if __name__ == "__main__":
    asyncio.run(main())
