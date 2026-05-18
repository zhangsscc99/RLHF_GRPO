#!/usr/bin/env python3
"""Offline GRPO smoke runner for the migrated RL-master FastApply pipeline.

One built-in FastApply task and 8 sampled responses are enough to exercise the
72-image pipeline without vLLM/DeepSpeed: load prompt -> rollout n responses ->
rule/LLM-judge reward -> group-relative advantage -> policy/KL/entropy loss.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rl_master.agent.environment.packages.fastapply import FastApply
from rl_master.agent.evaluation.extractors import extract_tagged_content
from rl_master.agent.trainer.grpo import GRPOConfig, RolloutSample, compute_grpo_loss, summarize_loss
from rl_master.applications.fastapply.reward.score import compute_score_by_rule

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "rl_master_fastapply_grpo_smoke.json"


@dataclass(frozen=True)
class Rollout:
    index: int
    text: str
    rule_correctness: float
    validity: float
    reward: float
    loss: dict[str, float]


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
        ["fastapply"] * len(responses),
        responses,
        [reference] * len(responses),
        [{"language": obs["language"]}] * len(responses),
    )

    samples: list[RolloutSample] = []
    validities: list[float] = []
    rewards: list[float] = []
    for i, (response, ok) in enumerate(zip(responses, correctness)):
        tagged = extract_tagged_content(response, tag="updated")
        candidate = tagged if tagged is not None else response
        valid = 1.0 if tagged is not None and bool(candidate.strip()) else 0.0
        reward = 0.8 * ok + 0.2 * valid
        validities.append(valid)
        rewards.append(reward)

        # Deterministic pseudo logprobs emulate the scalar means carried by
        # verl/MinT tensors for old actor, current actor, reference and entropy.
        old_lp = -0.45 - 0.03 * i
        new_lp = old_lp + (0.05 if ok else -0.02)
        ref_lp = -0.50 - 0.02 * i
        samples.append(
            RolloutSample(
                prompt_id="tiny-add",
                traj_group_id="tiny-add",
                response=response,
                reward=reward,
                old_logprob=old_lp,
                new_logprob=new_lp,
                ref_logprob=ref_lp,
                entropy=max(0.0, -new_lp),
                token_count=max(1, len(response.split())),
                metadata={"index": i, "rule_correctness": ok, "validity": valid},
            )
        )

    grpo_cfg = GRPOConfig(
        reward_grouping="traj_group_id",
        reward_normalization="mean_std",
        clip_ratio=0.2,
        adv_clip=2.0,
        use_kl_loss=True,
        kl_loss_coef=0.02,
        kl_loss_type="low_var_kl",
        entropy_coeff=0.001,
    )
    losses = compute_grpo_loss(samples, grpo_cfg)
    rollouts = [
        Rollout(
            index=int(sample.metadata["index"]),
            text=sample.response,
            rule_correctness=float(sample.metadata["rule_correctness"]),
            validity=float(sample.metadata["validity"]),
            reward=sample.reward,
            loss=loss.to_json(),
        )
        for sample, loss in zip(samples, losses)
    ]

    report: dict[str, Any] = {
        "pipeline": "RL-master FastApply GRPO smoke: data -> rollout(n=8) -> reward -> group advantage -> policy/KL/entropy loss",
        "model_target": "Qwen/Qwen3-4B-Instruct-2507 for MinT; offline smoke uses deterministic sampler",
        "config": {**asdict(grpo_cfg), "rollout_n": 8, "sequence_length": 8192, "temperature": 0.99, "top_p": 0.99, "top_k": 50},
        "prompt_observation_keys": sorted(obs.keys()),
        "reward_formula": "reward = 0.8 * rule_or_judge_correctness + 0.2 * valid_updated_output",
        "advantage_formula": "adv_i = (reward_i - mean(group_rewards)) / std(group_rewards)",
        "loss_formula": "total = clipped_policy_surrogate + beta_kl * low_var_KL(policy||ref) - entropy_coef * entropy",
        "mean_reward": mean(rewards),
        "loss_summary": summarize_loss(losses),
        "rollouts": [asdict(r) for r in rollouts],
        "image_reference_count": len(list((ROOT / "images").glob("*.jpg"))) + len(list((ROOT / "images").glob("*.png"))),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
