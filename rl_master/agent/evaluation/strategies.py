from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np


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
        candidate = sample.get(self._candidate_key, "")
        reference = sample.get(self._reference_key, "")
        ratio = 1.0 if candidate == reference else 0.0
        return EvaluationResult(ratio >= self._threshold, {"ratio": ratio})
