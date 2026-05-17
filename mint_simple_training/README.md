# MinT simple SFT smoke test

Run with Python 3.11+ and `MINT_API_KEY` in the environment:

```bash
export MINT_API_KEY='sk-...'
export MINT_BASE_URL='https://mint.macaron.xin/'
./.venv/bin/python simple_mint_sft.py
```

The script performs a one-step LoRA SFT smoke test on `Qwen/Qwen3-0.6B` using two tiny examples, saves a redacted JSON report to `training_report.json`, and does not print or persist the API key.
