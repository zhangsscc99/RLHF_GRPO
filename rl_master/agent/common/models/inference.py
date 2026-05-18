from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .chat.prompt import PromptTemplate
from .llm import ControlledLLM, GenerationConfig, LLM


class Inference:
    def __init__(self, model: LLM | ControlledLLM, template: PromptTemplate, parse: Optional[Callable[[str], Any]] = None, config: Optional[GenerationConfig] = None):
        self._model = model
        self._template = template
        self._parse = parse or (lambda x: x)
        self._config = config or GenerationConfig()

    async def __call__(self, data: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None) -> Any:
        messages = self._template.render(data)
        if history:
            messages = history + messages
        result = await self._model.chat(messages, **self._config.to_dict())
        if result is not None:
            if isinstance(result, list):
                if self._config.num == 1:
                    return self._parse(result[0]) if result else None
                return [self._parse(x) for x in result]
            return self._parse(result)
        return None

    async def close(self):
        if isinstance(self._model, ControlledLLM):
            await self._model.close()
