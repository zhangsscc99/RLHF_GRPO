from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_master.agent.environment.packages.fastapply import FastApply
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


if __name__ == "__main__":
    test_reward_rule_exact()
    test_fastapply_env_step()
    test_patch_build_and_compare()
    print("rl_master_fastapply tests passed")
