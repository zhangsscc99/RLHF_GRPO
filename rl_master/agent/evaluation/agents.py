from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from rl_master.agent.common.models import GenerationConfig, Inference, LLM, PromptTemplate


class Agent(ABC):
    @abstractmethod
    async def run(self, data: Dict[str, Any]):
        pass


class SimpleQA(Agent):
    def __init__(self, model: LLM, template: Optional[PromptTemplate] = None, parse: Optional[Callable[[Optional[str]], Any]] = None, config: Optional[GenerationConfig] = None):
        self._inference = Inference(model=model, template=template or PromptTemplate("{{ question }}", [PromptTemplate.Variable("question")]), parse=parse, config=config)

    async def run(self, data: Dict[str, Any]):
        return await self._inference(data)
