from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np

from rl_master.agent.common.models import GenerationConfig, Inference, LLM, PromptTemplate, RespondWithJSON
from rl_master.agent.evaluation.utils import visit


class Metric:
    def __init__(self, name: str, mode: Literal["mean", "sum", "max", "min"] = "mean"):
        self._name = name
        self._mode = mode
        self._history: List[float] = []

    @property
    def name(self) -> str:
        return self._name

    def add(self, value: int | float):
        self._history.append(float(value))

    @property
    def value(self) -> float:
        if not self._history:
            return 0.0
        match self._mode:
            case "sum": return float(np.sum(self._history))
            case "max": return float(np.max(self._history))
            case "min": return float(np.min(self._history))
            case _: return float(np.mean(self._history))


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@dataclass(frozen=True)
class EvaluationResult:
    is_correct: bool = field(metadata={"description": "sample is correct"})
    details: Optional[Dict[str, Any]] = None


class EvaluationStrategy:
    def __init__(self, pass_at_k: Optional[List[int]] = None):
        self._pass_at_k = pass_at_k or [1]
        self._pass_at_k_metrics = {f"pass@{k}": Metric(f"pass@{k}") for k in self._pass_at_k}

    @property
    def metrics(self) -> List[Metric]:
        return list(self._pass_at_k_metrics.values())

    async def evaluate(self, sample: Dict[str, Any]) -> Dict[str, float]:
        result = await self.__call__(sample)
        is_correct = bool(result.is_correct)
        for metric in self._pass_at_k_metrics.values():
            metric.add(1.0 if is_correct else 0.0)
        return {m.name: m.value for m in self.metrics}

    async def __call__(self, sample: Dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError


class EvaluationByExactMatch(EvaluationStrategy):
    def __init__(self, candidate_key: str = "candidate", reference_key: str = "reference", threshold: float = 1.0, pass_at_k: Optional[List[int]] = None):
        super().__init__(pass_at_k)
        self._candidate_key = candidate_key
        self._reference_key = reference_key
        self._threshold = threshold

    async def __call__(self, sample: Dict[str, Any]) -> EvaluationResult:
        candidate = visit(self._candidate_key, sample) if "." in self._candidate_key else sample.get(self._candidate_key, "")
        reference = visit(self._reference_key, sample) if "." in self._reference_key else sample.get(self._reference_key, "")
        ratio = 1.0 if candidate == reference else 0.0
        return EvaluationResult(ratio >= self._threshold, {"ratio": ratio})


class EvaluationByLLMJudge(EvaluationStrategy):
    """LLM-as-judge strategy mirroring the screenshots' EvaluationByJudge."""

    def __init__(
        self,
        model: Optional[LLM] = None,
        candidate_key: str = "candidate",
        reference_key: str = "reference",
        question_key: Optional[str] = None,
        pass_at_k: Optional[List[int]] = None,
    ):
        super().__init__(pass_at_k)
        self._candidate_key = candidate_key
        self._reference_key = reference_key
        self._question_key = question_key
        template = PromptTemplate(
            content=(
                "Candidate Answer: {{ candidate }}\n"
                "Reference Answer: {{ reference }}\n"
                "{% if question %}Question: {{ question }}{% endif %}\n"
                "Return JSON only: {\"judgement\":\"Yes\"} or {\"judgement\":\"No\"}."
            ),
            variables=[
                PromptTemplate.Variable("candidate"),
                PromptTemplate.Variable("reference"),
                PromptTemplate.Variable("question"),
            ],
            sys_content="You are a professional evaluator.",
        )
        self._inference = Inference(model or LLM("Qwen3-30B-A3B-Instruct"), template, parse=RespondWithJSON(), config=GenerationConfig(temperature=0.2))

    async def __call__(self, sample: Dict[str, Any]) -> EvaluationResult:
        data = {
            "candidate": visit(self._candidate_key, sample) if "." in self._candidate_key else sample.get(self._candidate_key),
            "reference": visit(self._reference_key, sample) if "." in self._reference_key else sample.get(self._reference_key),
            "question": visit(self._question_key, sample) if self._question_key and "." in self._question_key else sample.get(self._question_key or "", ""),
        }
        result = await self._inference(data)
        judgement = str((result or {}).get("judgement", "No")).lower() if isinstance(result, dict) else "no"
        return EvaluationResult(judgement in {"yes", "true", "pass"}, {"judge": result})


class EvaluationByTests(EvaluationStrategy):
    """Minimal hook for test-based evaluation.

    The image source has a test strategy placeholder.  Here the caller passes a
    function that returns True/False and optional details.
    """

    def __init__(self, test_fn: Callable[[Dict[str, Any]], bool], pass_at_k: Optional[List[int]] = None):
        super().__init__(pass_at_k)
        self._test_fn = test_fn

    async def __call__(self, sample: Dict[str, Any]) -> EvaluationResult:
        ok = bool(self._test_fn(sample))
        return EvaluationResult(ok, {"strategy": "tests"})
