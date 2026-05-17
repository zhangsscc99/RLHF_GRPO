# RLHF / MinT training experiments

This repo contains small MinT smoke tests and a minimal reproduction of the photographed FastApply pipeline.

## Existing simple SFT

`mint_simple_training/simple_mint_sft.py` runs a one-step MinT SFT sanity check.

## FastApply pipeline reproduction

The 18 images describe an AI coding Apply model: given source code plus a patch, generate the fully updated source. This repo reproduces that flow with tiny test data first.

```bash
# build two tiny source+patch -> updated-source records
./mint_simple_training/.venv/bin/python -m fastapply_pipeline.build_dataset

# run one MinT SFT smoke step on the supported 4B Qwen model
export MINT_API_KEY='sk-...'
export MINT_BASE_URL='https://mint.macaron.xin/'
./mint_simple_training/.venv/bin/python -m fastapply_pipeline.train_mint_fastapply
```

Default model: `Qwen/Qwen3-4B-Instruct-2507`. MinT docs list this 4B Qwen3 model; no Qwen3.5 4B identifier was found in the supported model page.
