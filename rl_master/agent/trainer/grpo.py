from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence


AdvantageMethod = Literal["mean", "mean_std"]
KLLossType = Literal["low_var_kl", "kl"]


@dataclass(frozen=True)
class RolloutSample:
    """One generated response in a GRPO group.

    The screenshots show verl-style fields carried through the batch:
    group id, reward, old actor logprob, current actor logprob, ref logprob,
    entropy and token weights.  This tiny class is deliberately scalar so the
    offline smoke can run without torch while still preserving the same math.
    """

    prompt_id: str
    traj_group_id: str
    response: str
    reward: float
    old_logprob: float
    new_logprob: float
    ref_logprob: float
    entropy: float
    token_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GRPOConfig:
    adv_estimator: str = "grpo"
    reward_grouping: str = "traj_group_id"
    reward_normalization: AdvantageMethod = "mean_std"
    whiten_advantages: bool = True
    adv_clip: float | None = 2.0
    clip_ratio: float = 0.2
    dual_clip_loss: bool = True
    dual_clip_lower_bound: float = 3.0
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.05
    kl_loss_type: KLLossType = "low_var_kl"
    entropy_coeff: float = 0.001


@dataclass(frozen=True)
class GRPOLossBreakdown:
    prompt_id: str
    traj_group_id: str
    reward: float
    advantage: float
    ratio: float
    clipped_ratio: float
    policy_loss: float
    kl_loss: float
    entropy_loss: float
    total_loss: float
    token_count: int

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _group(samples: Iterable[RolloutSample], key: str) -> Dict[str, List[RolloutSample]]:
    groups: Dict[str, List[RolloutSample]] = defaultdict(list)
    for sample in samples:
        groups[str(getattr(sample, key))].append(sample)
    return dict(groups)


def compute_group_advantages(samples: Sequence[RolloutSample], config: GRPOConfig | None = None) -> Dict[int, float]:
    """Compute GRPO group-relative advantages.

    Images 11-13 show the core formula:
    ``advantage_i = (reward_i - mean(reward_group)) / std(reward_group)``.
    When the group has no variance we keep the centered value at 0 instead of
    injecting noise.  Optional clipping mirrors the photographed adv_clip.
    """

    cfg = config or GRPOConfig()
    result: Dict[int, float] = {}
    for _, group in _group(samples, cfg.reward_grouping).items():
        rewards = [float(s.reward) for s in group]
        group_mean = mean(rewards) if rewards else 0.0
        group_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        for sample in group:
            adv = float(sample.reward) - group_mean
            if cfg.reward_normalization == "mean_std" and group_std > 1e-12:
                adv /= group_std
            if cfg.adv_clip is not None:
                adv = _clip(adv, -cfg.adv_clip, cfg.adv_clip)
            result[id(sample)] = adv
    return result


def low_var_kl(actor_logprob: float, ref_logprob: float) -> float:
    """verl low-variance KL: exp(ref - actor) - (ref - actor) - 1."""

    delta = float(ref_logprob) - float(actor_logprob)
    return math.exp(delta) - delta - 1.0


def _plain_kl(actor_logprob: float, ref_logprob: float) -> float:
    return float(actor_logprob) - float(ref_logprob)


def _policy_loss(ratio: float, advantage: float, cfg: GRPOConfig) -> tuple[float, float]:
    clipped_ratio = _clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
    surrogate = ratio * advantage
    clipped_surrogate = clipped_ratio * advantage
    if cfg.dual_clip_loss and advantage < 0:
        # PPO dual-clip guard for negative advantages; keeps very large ratios
        # from over-penalising one bad sample.
        lower = cfg.dual_clip_lower_bound * advantage
        chosen = max(min(surrogate, clipped_surrogate), lower)
    else:
        chosen = min(surrogate, clipped_surrogate)
    return -chosen, clipped_ratio


def compute_grpo_loss(samples: Sequence[RolloutSample], config: GRPOConfig | None = None) -> List[GRPOLossBreakdown]:
    """Compute the scalar loss decomposition shown in the screenshots.

    This is not a replacement for torch/verl; it is an auditable, dependency-free
    reproduction of the same policy / KL / entropy accounting for smoke tests.
    """

    cfg = config or GRPOConfig()
    advantages = compute_group_advantages(samples, cfg)
    rows: List[GRPOLossBreakdown] = []
    for sample in samples:
        adv = advantages[id(sample)]
        ratio = math.exp(float(sample.new_logprob) - float(sample.old_logprob))
        policy_loss, clipped_ratio = _policy_loss(ratio, adv, cfg)
        if cfg.use_kl_loss:
            raw_kl = low_var_kl(sample.new_logprob, sample.ref_logprob) if cfg.kl_loss_type == "low_var_kl" else _plain_kl(sample.new_logprob, sample.ref_logprob)
            kl_loss = cfg.kl_loss_coef * raw_kl
        else:
            kl_loss = 0.0
        entropy_loss = -cfg.entropy_coeff * float(sample.entropy)
        total = policy_loss + kl_loss + entropy_loss
        rows.append(
            GRPOLossBreakdown(
                prompt_id=sample.prompt_id,
                traj_group_id=sample.traj_group_id,
                reward=sample.reward,
                advantage=adv,
                ratio=ratio,
                clipped_ratio=clipped_ratio,
                policy_loss=policy_loss,
                kl_loss=kl_loss,
                entropy_loss=entropy_loss,
                total_loss=total,
                token_count=max(1, sample.token_count),
            )
        )
    return rows


def summarize_loss(rows: Sequence[GRPOLossBreakdown]) -> Dict[str, float]:
    if not rows:
        return {"loss/mean": 0.0}
    fields = ["reward", "advantage", "ratio", "policy_loss", "kl_loss", "entropy_loss", "total_loss"]
    return {f"{field}/mean": mean(float(getattr(row, field)) for row in rows) for field in fields}
