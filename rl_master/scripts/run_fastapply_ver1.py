#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from rl_master.agent.environment.packages.fastapply import FastApply
from rl_master.applications.fastapply.reward.score import compute_score_by_rule


def main() -> int:
    env = FastApply()
    obs = env.reset()
    candidate = "<updated>\ndef add(a, b):\n    return a + b + 1\n</updated>"
    _, reward, done, _, info = env.step(candidate)
    score = compute_score_by_rule(["fastapply"], [candidate], ["def add(a, b):\n    return a + b + 1\n"], [{"language": "python"}])[0]
    report = {"obs_keys": sorted(obs), "env_reward": reward, "rule_score": score, "done": done, "info": info}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/rl_master_fastapply_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
