# FastApply pipeline reproduction notes

This project recreates the 18-image FastApply training pipeline as code, as the actual GRPO pipeline.

## Page-by-page mapping from the images

1. **AI Coding Apply task**: full generation and incremental generation; planner/editor/verifier loop; intent parsing -> plan -> locate -> precise patch -> validate.
2. **Apply scenarios**: whole-file update is wasteful; Search/Replace can fail on formatting/context; Apply model should understand intent, structure, and context.
3. **FastApply development**: use specialized Apply data and model; maintain original code structure and output updated code.
4. **Task prompt**: coding assistant specialized in merging code updates; output within `<updated>` tags only.
5. **Template**: language + source + patch; patch can be SEARCH/REPLACE hunks, multiple hunks, or structured updates.
6-8. **Metrics/deployment**: FastApply Qwen 4B variant, token/sec benchmarks, vLLM deployment notes.
9. **Inference service**: endpoint accepts source/patch/language and returns updated content.
10-11. **Data construction**: mine GitPython commit/diff, keep commit hash/language/old_source/new_source/diff; parse structured changes; filter, dedupe, build train examples.
12-13. **Edit correction prompts**: for failed Search/Replace, generate corrected search strings with strict rules.
14-15. **AST analysis**: identify top-level scopes for Python/Java/JS and keep edits inside meaningful boundaries.
16-17. **Serving and RL**: FastApply service endpoint; PPO-like experiments, GRPO/GSPO mentioned, final training mainly GRPO.
18. Duplicate/preview of page 1.

## Implemented modules

- `test_data.py`: two tiny Apply examples only. No large synthetic dataset is fabricated.
- `patching.py`: SEARCH/REPLACE and structured-changes replay, unified-diff helper.
- `filters.py`: size filtering, patch replay validation, syntax/AST check, dedup hash.
- `ast_tools.py`: Python AST and JS top-level scope extraction placeholder.
- `templates.py`: photographed `<language>`, `<source>`, `<patch>`, `<updated>` prompt contract.
- `rewards.py`: FastApply verifier reward: exact updated source, similarity, tag format, no extra prose, syntax validity.
- `grpo.py`: group reward centering and MinT `types.Datum` construction with `weights`, `logprobs`, and `advantages`.
- `train_mint_fastapply_grpo.py`: actual MinT GRPO loop using `loss_fn="importance_sampling"`.
- `train_mint_fastapply.py`: legacy warmup path removed; GRPO is the reproduced pipeline.

## GRPO flow

1. Save current LoRA weights and create a sampler.
2. Sample multiple candidate updates for the same `source + patch` prompt.
3. Score every sample with the FastApply verifier.
4. Compute group-relative advantages: `advantage = reward - mean(group_rewards)`.
5. Build MinT datums with response-token weights, sampled logprobs, and advantages.
6. Train with `training_client.forward_backward(datums, loss_fn="importance_sampling")`.
7. `optim_step`, save weights, sample once for verification.

For tiny smoke stability, the script can add one oracle rollout after policy sampling via `--include-oracle-rollout`; this keeps the GRPO advantage signal non-degenerate with only one test prompt.
