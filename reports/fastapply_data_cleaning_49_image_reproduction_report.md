# FastApply 数据清洗 49 张微信图复刻报告

## 结论

已按 `/root/rlhf/images/微信图片_20260519213820_348_138.jpg` 到 `微信图片_20260519213908_396_138.jpg` 的 49 张截图，把 RL-master/FastApply 的 **数据清洗与构建 pipeline** 迁移到本项目：

- 数据结构层：`Span`、`Hunk`、`Unit`、`UnitState`、`ChangeContext`
- AST 层：Python / Java / fallback 语法单元树，兼容无 tree-sitter 的服务器环境
- 清洗规则层：import 修改、functional 修改、function/class scope、hunk 数、长度、语法检查
- build CLI 层：archive / extract / parse / filter / deduplicate / generate / make / convert / analyze
- tiny 可跑数据：`data/fastapply_cleaning_tiny_raw.jsonl`
- smoke test：`tests/test_fastapply_data_cleaning.py`

验证命令：

```bash
./mint_simple_training/.venv/bin/python -m compileall -q rl_master tests fastapply_pipeline
./mint_simple_training/.venv/bin/python tests/test_fastapply_data_cleaning.py
```

额外跑通了一条本地数据构建链：

```bash
./mint_simple_training/.venv/bin/python -m rl_master.applications.fastapply.build parse parse \
  --dataset-load-path data/fastapply_cleaning_tiny_raw.jsonl \
  --dataset-save-path /tmp/fastapply_cleaning_smoke/parsed.jsonl
./mint_simple_training/.venv/bin/python -m rl_master.applications.fastapply.build make generate \
  --dataset-load-path /tmp/fastapply_cleaning_smoke/parsed.jsonl \
  --dataset-save-path /tmp/fastapply_cleaning_smoke/generated.jsonl
./mint_simple_training/.venv/bin/python -m rl_master.applications.fastapply.build generate clean \
  --dataset-load-path /tmp/fastapply_cleaning_smoke/generated.jsonl \
  --dataset-save-path /tmp/fastapply_cleaning_smoke/cleaned.jsonl
./mint_simple_training/.venv/bin/python -m rl_master.applications.fastapply.build convert convert \
  --dataset-load-path /tmp/fastapply_cleaning_smoke/cleaned.jsonl \
  --dataset-save-path /tmp/fastapply_cleaning_smoke/prompts.jsonl
```

## 49 张图对应的源码文件

| 图片范围 | 复刻文件 | 核心内容 |
|---|---|---|
| 1-6、7-12、13-14 | `rl_master/applications/fastapply/data/types.py`、`data/utils.py` | Span 行范围运算；Hunk 解析 unified diff；Unit/UnitState/ChangeContext；DFS/BFS、DCA、context 下探和扩展。 |
| 15 | `rl_master/applications/fastapply/data/ops.py` | Repo 配置、clone/reuse repo。 |
| 16、27 | `rl_master/applications/fastapply/data/__init__.py`、`data/ast/__init__.py` | AST parser 注册、语言检测、导出。 |
| 17-26 | `rl_master/applications/fastapply/data/ast/*` | AbstractSyntaxTree、Python AST、Java AST、parser registry、debug_ast。 |
| 28-31 | `rl_master/applications/fastapply/build/parse.py` | 从 git diff 提取 hunk，构造成 SEARCH/REPLACE change blocks，并支持 train/test split。 |
| 32-34 | `rl_master/applications/fastapply/build/make.py` | 用 changes 构造 patch；截图原版走 LLM，这里保留接口，同时 tiny 数据用 deterministic `new_source` fallback。 |
| 35-39 | `rl_master/applications/fastapply/build/generate.py`、`build/filter.py` | AST + 规则过滤：import 修改、functional 修改、hunk 数、长度、语法。 |
| 40-42 | `rl_master/applications/fastapply/build/extract.py`、`build/filter.py` | git commit diff 抽取、文件后缀筛选、blob old/new source 获取。 |
| 43-44 | `rl_master/applications/fastapply/build/deduplicate.py` | MinHash 思路的去重；当前用轻量 hash signature fallback。 |
| 45 | `rl_master/applications/fastapply/build/convert.py` | 转成 GRPO prompt / reward_model / extra_info 数据格式。 |
| 46-47 | `rl_master/applications/fastapply/build/archive.py` | repo 文件夹 zip 压缩/解压。 |
| 48 | `rl_master/applications/fastapply/build/analyze.py` | token 长度桶分析。 |
| 49 | `rl_master/applications/fastapply/build/__cli__.py`、`build/__main__.py` | Typer CLI 入口，汇总所有 build 子命令。 |

## 数据清洗 pipeline

1. **archive/decompress**：先把很多代码仓库打包/解包，便于离线批处理。
2. **extract**：遍历 Git 仓库单父提交，取每个 commit 的 source-code 文件 diff；同时读取 `old_source` 与 `new_source`。
3. **parse**：把 unified diff 切成 `Hunk`，每个 hunk 构造成 `{old, new}` 的 SEARCH/REPLACE change。
4. **AST 建树**：把 old/new source 解析为 `Unit` 树；Python 用 stdlib `ast`，Java 用 brace/regex fallback。
5. **Hunk -> Unit 映射**：`ChangeContext('-', old_units, hunks)` 和 `ChangeContext('+', new_units, hunks)` 把修改行号映射到 AST Unit。
6. **context 扩展**：`expand_context_to_all_units` 把 top-level 的修改状态扩展到函数、类、语句等子节点。
7. **规则过滤**：保留满足数据目标的样本：有 functional 修改，截图规则可要求 import 修改；过滤空源码、过多 hunk、过长样本、语法失败样本。
8. **make/generate**：截图原版用 LLM 根据 SEARCH/REPLACE 生成 update；本项目 tiny 路径直接使用 `new_source`，保证可跑。
9. **deduplicate**：按语言和 update signature 去重，避免同质样本刷高训练 reward。
10. **convert**：转成 GRPO 训练行：`prompt`、`reward_model.ground_truth`、`extra_info.language/file`、`response=<updated>...</updated>`。
11. **analyze/split**：按 token bucket 分析，再拆 train/test。

## 为什么这样适配

截图依赖 tree-sitter、GitPython、datasets/parquet、datasketch/MinHash、LLM 服务等。本服务器当前训练环境未安装这些重依赖，所以复刻时做了兼容：

- 保留原始文件名、类名、函数名和调用链。
- 有依赖时可替换接入；无依赖时用轻量 fallback。
- tiny 数据只保留 1 条 import+function 修改样本，确保能在本项目 smoke 跑通。
- 清洗产物最后接到已有 FastApply GRPO prompt/reward 格式，和之前 minT/Qwen3 4B GRPO 训练代码保持一致。
