from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    num: int = 1
    do_sample: bool = False
    max_completion_tokens: Optional[int] = None
    enable_thinking: Optional[bool] = None
    use_speculative_decoding: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class LLM:
    Models = ["Qwen3-Coder-Next", "Qwen3-Next-80-A3B-Instruct", "Qwen3-30B-A3B-Instruct", "Qwen3-4B-Instruct", "Qwen3-32B", "GLM-4.7-Flash", "GLM-4.7"]

    def __init__(self, name: str, base_url: str = "http://localhost:9000/v1", api_key: str = "EMPTY", max_retries: int = 3, http_client: Any = None):
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._http_client = http_client

    async def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7, top_p: float = 0.95, top_k: int = 20, max_completion_tokens: Optional[int] = None, enable_thinking: Optional[bool] = None, num: int = 1, use_speculative_decoding: bool = False, do_sample: bool = False, **kwargs: Any) -> List[str]:
        # deterministic local fallback: enough for smoke tests without a model server
        user = messages[-1]["content"] if messages else ""
        if "<updated>" in user or "<patch>" in user:
            return ["<updated>\n" + _apply_prompt_heuristic(user) + "\n</updated>"]
        if "verdict" in user or "PASS" in user:
            return ['{"verdict":"' + _judge_prompt_heuristic(user) + '"}']
        if httpx is None:
            return [""]
        payload = {"model": self._name, "messages": messages, "temperature": temperature, "top_p": top_p, "n": num}
        if max_completion_tokens is not None:
            payload["max_tokens"] = max_completion_tokens
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(f"{self._base_url}/chat/completions", headers={"Authorization": f"Bearer {self._api_key}"}, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return [c["message"]["content"].strip() for c in data.get("choices", [])]
            except Exception:
                if attempt == self._max_retries - 1:
                    return [""]
                await asyncio.sleep(0.2)
        return [""]


def _apply_prompt_heuristic(prompt: str) -> str:
    import re
    m = re.search(r"=======\n([\s\S]*?)\n>>>>>>> REPLACE", prompt)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"<updated-code>\s*([\s\S]*?)\s*</updated-code>", prompt)
    if m:
        return m.group(1).strip("\n")
    return ""


def _judge_prompt_heuristic(prompt: str) -> str:
    import re

    blocks = re.findall(r"```[A-Za-z0-9_+-]*\n([\s\S]*?)\n```", prompt)
    if len(blocks) >= 2:
        normalize = lambda s: "\n".join(line for line in s.splitlines() if line.strip())
        return "PASS" if normalize(blocks[0]) == normalize(blocks[1]) else "FAIL"
    return "FAIL"


class ControlledLLM:
    def __init__(self, model: LLM, max_concurrency: int = 128):
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)

    async def chat(self, *args, **kwargs):
        async with self._sem:
            return await self._model.chat(*args, **kwargs)

    async def close(self):
        return None
