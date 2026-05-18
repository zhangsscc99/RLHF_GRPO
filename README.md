# RLHF / MinT training experiments

This repo contains MinT smoke tests and a code-level reproduction of the photographed FastApply pipeline.

## FastApply pipeline reproduction

The images describe a coding Apply model: given source code plus a patch, generate the fully updated source. The reproduced training pipeline here is GRPO-based.

## RL-master 72-image migration

The 72 screenshots under `/root/rlhf/images` were migrated into the `rl_master/`
package as a runnable RL-master / verl-style FastApply GRPO stack:

```bash
./mint_simple_training/.venv/bin/python -m rl_master.scripts.run_fastapply_ver1
./mint_simple_training/.venv/bin/python -m rl_master.scripts.run_fastapply_grpo_smoke
./mint_simple_training/.venv/bin/python -m rl_master.scripts.evaluation.evalfastapply
./mint_simple_training/.venv/bin/python tests/test_rl_master_fastapply.py
```

The full image-by-image migration table is in
`reports/rl_master_72_image_reproduction_report.md`.

Build the tiny test dataset:

```bash
./mint_simple_training/.venv/bin/python -m fastapply_pipeline.build_dataset
```

Run the GRPO reproduction on MinT Qwen3 4B:

```bash
export MINT_API_KEY='sk-...'
export MINT_BASE_URL='https://mint.macaron.xin/'
export MINT_BASE_MODEL='Qwen/Qwen3-4B-Instruct-2507'
./mint_simple_training/.venv/bin/python -m fastapply_pipeline.train_mint_fastapply_grpo --group-size 2 --include-oracle-rollout
```

Default model: `Qwen/Qwen3-4B-Instruct-2507`.

## Modules

- `patching.py`: SEARCH/REPLACE and structured patch replay.
- `filters.py`: replay validation, syntax/AST checks, dedupe.
- `ast_tools.py`: top-level scope analysis.
- `rewards.py`: FastApply verifier reward.
- `grpo.py`: group-relative advantages and `importance_sampling` Datum construction.
- `train_mint_fastapply_grpo.py`: MinT GRPO training loop.

The reproduced pipeline entrypoint is `fastapply_pipeline/train_mint_fastapply_grpo.py`.
