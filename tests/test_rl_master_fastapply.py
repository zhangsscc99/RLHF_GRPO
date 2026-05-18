from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapply_pipeline.ast_tools import analyze_source
from rl_master.agent.environment.packages.fastapply import FastApply
from rl_master.agent.trainer.grpo import GRPOConfig, RolloutSample, compute_grpo_loss
from rl_master.applications.fastapply.reward.score import compute_score_by_rule
from rl_master.applications.fastapply.reward.utils import build, compare
from rl_master.agent.evaluation.extractors import extract_tagged_content


def test_reward_rule_exact():
    output = "<updated>\ndef x():\n    return 1\n</updated>"
    assert compute_score_by_rule(["x"], [output], ["def x():\n    return 1\n"], [{"language": "python"}]) == [1.0]


def test_fastapply_env_step():
    env = FastApply()
    env.reset()
    _, reward, done, _, info = env.step("<updated>\ndef add(a, b):\n    return a + b + 1\n</updated>")
    assert done is True
    assert reward == 1.0
    assert info["metrics"]["success"] is True


def test_patch_build_and_compare():
    patch = build([{"old": "a\n", "new": "b\n"}])
    assert "SEARCH" in patch and "REPLACE" in patch
    assert compare("a\n", "a\n") is True
    assert extract_tagged_content("<updated>x</updated>", "updated") == "x"


def test_grpo_loss_group_advantage():
    samples = [
        RolloutSample("p", "g", "good", 1.0, -0.5, -0.45, -0.5, 0.45, 4),
        RolloutSample("p", "g", "bad", 0.0, -0.5, -0.55, -0.5, 0.55, 4),
    ]
    rows = compute_grpo_loss(samples, GRPOConfig(kl_loss_coef=0.02, entropy_coeff=0.001))
    assert rows[0].advantage > 0
    assert rows[1].advantage < 0
    assert rows[0].policy_loss < rows[1].policy_loss


def test_multilanguage_syntax_analysis():
    assert analyze_source("python", "def ok():\n    return 1\n").parse_ok is True
    assert analyze_source("python", "def bad(:\n").parse_ok is False
    assert analyze_source("json", '{"a": 1}').parse_ok is True
    assert analyze_source("json", '{"a": }').parse_ok is False
    # Common languages now go through local checkers when installed, otherwise
    # a marked heuristic fallback instead of being hardcoded to JS/Python only.
    js_info = analyze_source("typescript", "function ok(): number { return 1 }\n")
    assert js_info.language == "typescript"
    assert js_info.checker


if __name__ == "__main__":
    test_reward_rule_exact()
    test_fastapply_env_step()
    test_patch_build_and_compare()
    test_grpo_loss_group_advantage()
    test_multilanguage_syntax_analysis()
    print("rl_master_fastapply tests passed")
