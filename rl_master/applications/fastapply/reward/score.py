from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import compare, generate_diff_view
from rl_master.agent.common.models import GenerationConfig, Inference, LLM, PromptTemplate, RespondWithJSON
from rl_master.agent.evaluation.extractors import extract_tagged_content


def _load_template() -> PromptTemplate:
    path = Path(__file__).resolve().parents[3] / "prompts" / "judge.yaml"
    return PromptTemplate.from_file(path)


async def compute_score_on_batch(languages: List[str], references: List[str], candidates: List[str]) -> List[float]:
    model = LLM("Qwen3-30B-A3B-Instruct", "http://localhost:9000/v1")
    template = _load_template()
    judge = Inference(model=model, template=template, parse=RespondWithJSON(), config=GenerationConfig(temperature=0.2, max_completion_tokens=8192))

    async def process(language: str, reference: str, candidate: str) -> float:
        if not candidate:
            return 0.0
        if compare(reference, candidate):
            return 1.0
        data = {"language": language, "reference": reference, "candidate": candidate, "diff": generate_diff_view(reference, candidate, language=language if language in {"python", "java", "javascript", "typescript"} else "python")}
        judgement: Optional[Dict[str, Any]] = await judge(data)
        if isinstance(judgement, list):
            judgement = judgement[0] if judgement else None
        if judgement is None:
            return 0.0
        verdict = judgement.get("verdict", "FAIL")
        return 1.0 if verdict == "PASS" else 0.0
    return await _gather(process, languages, references, candidates)


async def _gather(process, languages, references, candidates):
    tasks = [process(language, reference, candidate) for language, reference, candidate in zip(languages, references, candidates)]
    return await asyncio.gather(*tasks)


def compute_score(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[float]:
    candidates = [extract_tagged_content(solution_str, tag="updated") or "" for solution_str in solution_strs]
    references = ground_truths
    languages = [(extra_info or {}).get("language", "python") for extra_info in (extra_infos or [{} for _ in references])]
    try:
        return asyncio.run(compute_score_on_batch(languages, references, candidates))
    except RuntimeError:
        return compute_score_by_rule(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)


def compute_score_by_rule(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[float]:
    candidates = [extract_tagged_content(solution_str, tag="updated") or "" for solution_str in solution_strs]
    return [1.0 if compare(reference, candidate) else 0.0 for reference, candidate in zip(ground_truths, candidates)]
