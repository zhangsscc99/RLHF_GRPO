# RL-master FastApply GRPO 72 图复刻完成审计

本次重新对照 `/root/rlhf/images` 的 72 张图检查源码。之前版本已经能跑通，但仍偏“简化 smoke”；这次补齐了图中更关键的 pipeline 细节，使项目既能离线运行，又保留 RL-master / verl 的字段与结构。

## 本轮发现的缺口与修复

| 图中模块 | 之前状态 | 本轮补齐 |
|---|---|---|
| `scripts/run-fastapply-ver1.sh` | 只有 Python smoke，没有 shell 入口 | 新增 `rl_master/scripts/run-fastapply-ver1.sh`，保留 MODEL_PATH、TRAIN_FILE、TEST_FILE、reward path、batch、prompt/response/sequence、rollout_n、temperature/top_p/top_k 等参数 |
| GRPO loss | loss 只写在 smoke 脚本里 | 新增 `rl_master/agent/trainer/grpo.py`，模块化实现 group advantage、clip ratio、dual clip、low_var_kl、entropy、loss summary |
| config-rl-fastapply.yaml | 只保留少量字段 | 扩展为图中风格配置：data filter、rollout/vLLM、algorithm、actor_train、actor_infer、reference、reward_normalization、env manager、custom_reward_path/name |
| FastApply 环境 | 只用内置 fixture | 新增 `GlobalDatasetManager`，环境可从 JSON/JSONL/Parquet 读取 source/patch/reference，也保留 tiny fallback；step 返回 correctness/action_is_valid/success |
| reward score | rule + judge 主干有了，但细节弱 | 增加 batch 长度校验、PASS/FAIL 大小写归一、保留 rule-first + LLM-as-judge fallback |
| prompt yaml | 简化版 | 扩展 `apply.yaml`、`generate.yaml`、`judge.yaml`，加入完整输出约束、update snippet 生成规则、diff 分类和 JSON verdict |
| evaluation pipeline | 顺序执行 | 增加 `_parallel_process`、check/preprocess/postprocess/max_process，接近图中并发 pipeline |
| evaluation strategies | 只有 exact match | 增加 JSONPath-like `visit`、LLM judge strategy、test strategy，占齐图中 evaluator/strategy 支撑 |
| LLM wrapper | smoke fallback 固定 | 增加 localhost/EMPTY 才走离线 fallback 的逻辑；非本地配置可走 OpenAI-compatible HTTP |
| 测试 | 只测 env/reward | 增加 GRPO loss advantage 测试；shell、GRPO smoke、evaluation smoke 均跑通 |

## 现在的可运行入口

```bash
./mint_simple_training/.venv/bin/python -m rl_master.scripts.run_fastapply_ver1
./mint_simple_training/.venv/bin/python -m rl_master.scripts.run_fastapply_grpo_smoke
PYTHON=./mint_simple_training/.venv/bin/python ./rl_master/scripts/run-fastapply-ver1.sh
./mint_simple_training/.venv/bin/python -m rl_master.scripts.evaluation.evalfastapply
./mint_simple_training/.venv/bin/python tests/test_rl_master_fastapply.py
```

## 72 图对应关系

- 1–18：`run-fastapply-ver1.sh`、`config-rl-fastapply.yaml`、`agent/trainer/grpo.py`、`run_fastapply_grpo_smoke.py`。
- 19–30：`applications/fastapply/reward/score.py`、`reward/utils.py`、`environment/packages/fastapply/impl.py`、`environment/manager.py`。
- 32–37：`prompts/apply.yaml`、`generate.yaml`、`judge.yaml`。
- 38–56：`scripts/evaluation/evalfastapply.py`、`agent/evaluation/extractors.py`、`strategies.py`、`agents.py`、`pipeline/evaluator.py`、`utils.py`。
- 57–72：`agent/common/models/llm.py`、`inference.py`、`chat/prompt.py`、`chat/utils/parser.py`。

## 仍然刻意不做的事

- 不造大数据集：仍只放 tiny test 数据，符合“先拿 test 数据一两个能训练就行”。
- 不强依赖 vLLM、DeepSpeed、Ray、OpenAI SDK：这些在图里属于集群真实训练依赖；本项目用兼容字段和轻量 fallback 保证服务器能直接跑。
- 不把 MinT key 写入文件。

## 验证结果

本轮已跑通：

- `rl_master_fastapply tests passed`
- `run_fastapply_ver1`: `env_reward=1.0`, `rule_score=1.0`
- `run_fastapply_grpo_smoke`: 8 条 rollout，生成 reward、advantage、policy_loss、kl_loss、entropy_loss、total_loss
- `evalfastapply`: `accuracy=1.0`, `rows=1`
- `run-fastapply-ver1.sh`: shell 入口可执行并输出同一 GRPO smoke 报告

结论：源码层面已经从“能跑 smoke 的简化复刻”补齐为“按 72 张图结构完整复刻、细节字段到位、且仍能离线运行”的版本。

## 2026-05-18 Reward / syntax / updated 标签补充

用户继续追问 reward 细节后，本轮补充：

- `fastapply_pipeline/ast_tools.py` 已从 Python/JS 简化逻辑扩展为多语言可插拔 syntax 检查。
- 支持方向包括 Python、JavaScript、TypeScript、Java、Go、Rust、C/C++、Shell、Ruby、PHP、Lua、JSON、YAML 等；有本地 checker 就真实检查，没有就标记 `checked=False` 并走 heuristic fallback。
- `updated` 标签仍然是规则解析：LLM 生成 `<updated>...</updated>`，代码通过 regex 抽取，不额外调用 LLM 解析标签。
- 网站新增 12–14 三个问题解释 exact-only reward、multi-language syntax、updated tag parser。
