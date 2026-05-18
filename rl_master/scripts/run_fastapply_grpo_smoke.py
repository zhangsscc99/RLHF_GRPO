#!/usr/bin/env python3
"""Offline GRPO smoke runner for the migrated RL-master FastApply pipeline.

This is intentionally tiny: one built-in FastApply task and 8 sampled responses so it
can run on this server without vLLM/DeepSpeed.  It mirrors the photographed pipeline:
load prompt -> rollout n responses -> rule/LLM-judge reward -> group relative
advantage -> policy/KL/entropy loss accounting -> report.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from rl_master.agent.environment.packages.fastapply import FastApply
from rl_master.applications.fastapply.reward.score import compute_score_by_rule
from rl_master.agent.evaluation.extractors import extract_tagged_content

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "rl_master_fastapply_grpo_smoke.json"


@dataclass(frozen=True)
class Rollout:
    index: int
    text: str
    rule_correctness: float
    validity: float
    reward: float
    advantage: float
    old_logprob: float
    new_logprob: float
    ref_logprob: float
    token_count: int
    policy_loss: float
    kl_loss: float
    entropy_bonus: float
    total_loss: float


def normalize_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    m = mean(rewards)
    s = pstdev(rewards) or 1.0
    return [(r - m) / s for r in rewards]


def candidate_rollouts(reference: str) -> list[str]:
    # n=8 follows the config/images; a tiny deterministic list replaces a live vLLM sampler.
    return [
        f"<updated>\n{reference}\n</updated>",
        "<updated>\ndef add(a, b):\n    return a + b\n</updated>",
        "<updated>\ndef add(a, b):\n    value = a + b + 1\n    return value\n</updated>",
        "def add(a, b):\n    return a + b + 1",
        "<updated>\ndef add(a, b):\n    return a + b + 2\n</updated>",
        "<updated>\ndef add(a, b):\n    # patched\n    return a + b + 1\n</updated>",
        "<updated>\n\n</updated>",
        f"<updated>\n{reference}\n</updated>",
    ]


def main() -> int:
    env = FastApply(seed=42, is_train=True)
    obs = env.reset()
    assert env._state is not None  # smoke-only access to the in-memory fixture
    reference = env._state.reference
    responses = candidate_rollouts(reference)
    correctness = compute_score_by_rule(
        ["fastapply"] * len(responses), responses, [reference] * len(responses), [{"language": obs["language"]}] * len(responses)
    )
    rewards = []
    validities = []
    for response, ok in zip(responses, correctness):
        tagged = extract_tagged_content(response, tag="updated")
        candidate = tagged if tagged is not None else response
        valid = 1.0 if tagged is not None and bool(candidate.strip()) else 0.0
        validities.append(valid)
        rewards.append(0.8 * ok + 0.2 * valid)
    advs = normalize_advantages(rewards)

    rollouts: list[Rollout] = []
    beta_kl = 0.02
    entropy_coef = 0.001
    clip_eps = 0.2
    for i, (text, ok, valid, reward, adv) in enumerate(zip(responses, correctness, validities, rewards, advs)):
        token_count = max(1, len(text.split()))
        # Deterministic pseudo logprobs emulate the fields carried by verl/MinT datums.
        old_lp = -0.45 - 0.03 * i
        new_lp = old_lp + (0.05 if ok else -0.02)
        ref_lp = -0.50 - 0.02 * i
        ratio = math.exp(new_lp - old_lp)
        clipped = min(max(ratio, 1 - clip_eps), 1 + clip_eps)
        policy_loss = -min(ratio * adv, clipped * adv)
        kl_loss = beta_kl * (math.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0)
        entropy_bonus = entropy_coef * max(0.0, -new_lp)
        total_loss = policy_loss + kl_loss - entropy_bonus
        rollouts.append(Rollout(i, text, ok, valid, reward, adv, old_lp, new_lp, ref_lp, token_count, policy_loss, kl_loss, entropy_bonus, total_loss))

    report: dict[str, Any] = {
        "pipeline": "RL-master FastApply GRPO smoke: data -> rollout(n=8) -> reward -> group advantage -> policy/KL/entropy loss",
        "model_target": "Qwen/Qwen3-4B-Instruct-2507 for MinT; offline smoke uses deterministic sampler",
        "config": {"rollout_n": 8, "sequence_length": 8192, "temperature": 0.99, "top_p": 0.99, "top_k": 50, "adv_estimator": "grpo"},
        "prompt_observation_keys": sorted(obs.keys()),
        "reward_formula": "reward = 0.8 * rule_or_judge_correctness + 0.2 * valid_updated_output",
        "advantage_formula": "adv_i = (reward_i - mean(group_rewards)) / std(group_rewards)",
        "loss_formula": "total = clipped_policy_surrogate + beta_kl * KL(policy||ref) - entropy_coef * entropy",
        "mean_reward": mean(rewards),
        "mean_loss": mean([r.total_loss for r in rollouts]),
        "rollouts": [asdict(r) for r in rollouts],
        "image_reference_count": len(list((ROOT / "images").glob("*.jpg"))) + len(list((ROOT / "images").glob("*.png"))),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
