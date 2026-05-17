from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mint import types

from .rewards import RewardBreakdown, score_fastapply_output


@dataclass(frozen=True)
class RolloutRecord:
    prompt_id: str
    text: str
    reward: RewardBreakdown
    advantage: float
    token_count: int

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["reward"] = self.reward.to_json()
        return data


def build_grpo_datums_for_group(
    *,
    prompt_id: str,
    prompt_tokens: list[int],
    response_token_groups: list[list[int]],
    response_logprob_groups: list[list[float]],
    response_texts: list[str],
    expected_source: str,
    language: str,
) -> tuple[list[types.Datum], list[RolloutRecord]]:
    rewards = [score_fastapply_output(txt, expected_source, language) for txt in response_texts]
    mean_reward = sum(r.total for r in rewards) / len(rewards) if rewards else 0.0
    advantages = [r.total - mean_reward for r in rewards]
    datums: list[types.Datum] = []
    rollout_records: list[RolloutRecord] = []
    prefix = len(prompt_tokens) - 1
    for resp_tokens, resp_logprobs, text, reward, adv in zip(
        response_token_groups, response_logprob_groups, response_texts, rewards, advantages
    ):
        if not resp_tokens:
            continue
        if len(resp_logprobs) != len(resp_tokens):
            resp_logprobs = (resp_logprobs + [0.0] * len(resp_tokens))[: len(resp_tokens)]
        full = prompt_tokens + resp_tokens
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full[:-1]),
                loss_fn_inputs={
                    "target_tokens": full[1:],
                    "weights": [0.0] * prefix + [1.0] * len(resp_tokens),
                    "logprobs": [0.0] * prefix + resp_logprobs,
                    "advantages": [0.0] * prefix + [adv] * len(resp_tokens),
                },
            )
        )
        rollout_records.append(
            RolloutRecord(
                prompt_id=prompt_id,
                text=text,
                reward=reward,
                advantage=round(adv, 6),
                token_count=len(resp_tokens),
            )
        )
    return datums, rollout_records
