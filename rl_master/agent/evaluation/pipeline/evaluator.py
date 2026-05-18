from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional

from rl_master.agent.evaluation.agents import Agent
from rl_master.agent.evaluation.strategies import EvaluationStrategy

try:
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover - the smoke path should not require external deps
    class Dataset(list):
        """Tiny datasets.Dataset-compatible fallback used for local smoke tests."""

        @classmethod
        def from_list(cls, rows):
            return cls(rows)


class InferencePipeline:
    def __init__(self, agent: Agent, check: Optional[Callable[[Dataset], bool]] = None, preprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, postprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
        self._agent = agent
        self._check = check
        self._preprocess = preprocess
        self._postprocess = postprocess

    async def __call__(self, dataset: Dataset) -> Dataset:
        rows = []
        for item in dataset:
            data = dict(item)
            if self._preprocess:
                data = self._preprocess(data)
            data["result"] = await self._agent.run(data)
            if self._postprocess:
                data = self._postprocess(data)
            rows.append(data)
        return Dataset.from_list(rows)


class Evaluator:
    def __init__(self, pipeline: InferencePipeline, strategy: EvaluationStrategy):
        self._pipeline = pipeline
        self._strategy = strategy

    async def run(self, dataset: Dataset):
        dataset = await self._pipeline(dataset)
        reports = []
        for item in dataset:
            reports.append(await self._strategy.evaluate(dict(item)))
        return reports
