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


async def _parallel_process(dataset: Dataset, function: Callable[[Dict[str, Any]], Any], max_process: int = 8) -> Dataset:
    sem = asyncio.Semaphore(max(1, max_process))

    async def run_one(item):
        async with sem:
            result = function(dict(item))
            if asyncio.iscoroutine(result):
                result = await result
            return result

    rows = await asyncio.gather(*(run_one(item) for item in dataset))
    return Dataset.from_list(list(rows))


class InferencePipeline:
    def __init__(
        self,
        agent: Agent,
        check: Optional[Callable[[Dataset], bool]] = None,
        preprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        postprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        max_process: int = 8,
    ):
        self._agent = agent
        self._check = check
        self._preprocess = preprocess
        self._postprocess = postprocess
        self._max_process = max_process

    async def __call__(self, dataset: Dataset) -> Dataset:
        if self._check is not None and not self._check(dataset):
            raise RuntimeError("The dataset does not meet the check condition")
        if self._preprocess is not None:
            dataset = await _parallel_process(dataset, self._preprocess, self._max_process)

        async def process(item: Dict[str, Any]):
            data = dict(item)
            data["result"] = await self._agent.run(data)
            return data

        dataset = await _parallel_process(dataset, process, self._max_process)
        if self._postprocess is not None:
            dataset = await _parallel_process(dataset, self._postprocess, self._max_process)
        return dataset


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
