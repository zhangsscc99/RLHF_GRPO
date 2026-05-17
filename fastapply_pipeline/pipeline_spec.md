# FastApply pipeline reproduction notes

This project recreates the 18-image FastApply training pipeline as a minimal, runnable MinT prototype.

## What the images describe

1. **Task**: AI coding Apply model. Given source code plus an update patch, output the fully updated source file.
2. **Apply formats**:
   - Full-file updated text.
   - Unified diff / patch text.
   - SEARCH/REPLACE hunks.
   - Structured edit operations (`Insert` / `Replace` / `Delete`).
   - AST-level changes.
3. **Prompt contract**: preserve code structure, comments and indentation; output only the updated source in `<updated>...</updated>` tags.
4. **Model target**: a Qwen 4B coding apply model, originally described as Qwen3-4B-FastApply; for MinT this prototype uses the supported `Qwen/Qwen3-4B-Instruct-2507` 4B model.
5. **Data pipeline**:
   - Pull commits/diffs from Git repositories.
   - Parse old source, new source, file language, commit hash, and diff/patch.
   - Convert diffs into structured changes or SEARCH/REPLACE hunks.
   - Filter bad samples: oversized files, ambiguous replacements, syntax/parsing failures, non-code files.
   - Deduplicate near-identical samples.
   - Optionally annotate AST top-level scopes for Python/Java.
   - Build SFT records where prompt is `source + patch`, response is updated source.
6. **Training/eval**:
   - SFT first: cross-entropy on response tokens only.
   - Optional RL/GRPO stage later: reward exact pass, parse success, diff correctness, no extra prose, tag correctness.
   - Smoke-test uses 1-2 synthetic test samples only, not a real mined dataset.

## Prototype scope

The code here does not fabricate a large dataset. It creates two tiny deterministic Apply examples so the MinT training path can run end-to-end. The pipeline modules are structured so real Git-mined data can later replace the test fixture.
