#!/usr/bin/env python3
"""FastApply GRPO reproduction on MinT.

This is the closest code-level replica of the photographed pipeline:
Git/patch-style Apply records -> sampled rollouts -> FastApply verifier rewards ->
group-relative advantages -> MinT loss_fn="importance_sampling" (GRPO).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mint
from mint import types

from fastapply_pipeline.build_dataset import SFT_PATH, main as build_dataset
from fastapply_pipeline.grpo import build_grpo_datums_for_group
from fastapply_pipeline.templates import build_response

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "fastapply_grpo_training_report.json"
BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
RUN_NAME = os.environ.get("MINT_RUN_NAME", "fastapply-grpo-qwen3-4b-tiny-2026-05-17")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_records() -> list[dict[str, Any]]:
    if not SFT_PATH.exists():
        build_dataset()
    return [json.loads(line) for line in SFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--include-oracle-rollout", action="store_true", help="Add one gold rollout for tiny smoke stability; policy samples are still collected first.")
    args = parser.parse_args()

    started = time.time()
    records = load_records()[:1]  # tiny smoke: one prompt group is enough to exercise GRPO
    report: dict[str, Any] = {
        "started_at": now_iso(),
        "pipeline": "FastApply GRPO: source+patch -> sampled updated source -> verifier rewards -> centered advantages -> importance_sampling",
        "base_url": os.environ.get("MINT_BASE_URL", "https://mint.macaron.xin/"),
        "base_model": BASE_MODEL,
        "algorithm": "GRPO / loss_fn=importance_sampling",
        "lora_rank": 16,
        "dataset_path": str(SFT_PATH),
        "dataset_examples_used": len(records),
        "group_size_policy_samples": args.group_size,
        "include_oracle_rollout": args.include_oracle_rollout,
        "api_key": "provided via MINT_API_KEY (redacted)",
    }
    try:
        if not os.environ.get("MINT_API_KEY"):
            raise RuntimeError("MINT_API_KEY is not set")
        print(f"[{now_iso()}] create training client {BASE_MODEL}", flush=True)
        service_client = mint.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=16,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"purpose": "fastapply-grpo-reproduction", "date": "2026-05-17"},
        )
        tokenizer = training_client.get_tokenizer()
        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"{RUN_NAME}-rollout")

        datums = []
        rollout_json = []
        for rec in records:
            prompt_tokens = tokenizer.encode(rec["prompt"])
            print(f"[{now_iso()}] sample group prompt={rec['id']} group_size={args.group_size}", flush=True)
            sample_res = sampling_client.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=args.group_size,
                sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.8, top_p=1.0, seed=17),
            ).result(timeout=900)
            response_tokens = [list(seq.tokens) for seq in sample_res.sequences]
            response_logprobs = [list(seq.logprobs or [0.0] * len(seq.tokens)) for seq in sample_res.sequences]
            response_texts = [tokenizer.decode(seq.tokens) for seq in sample_res.sequences]

            if args.include_oracle_rollout:
                # The images use verifier/reward-guided GRPO. For a 1-example smoke test,
                # include one known-good trajectory so advantages are non-degenerate.
                oracle_text = build_response(json.loads((ROOT / "data" / "fastapply_tiny_raw.jsonl").read_text(encoding="utf-8").splitlines()[0])["new_source"])
                oracle_tokens = tokenizer.encode(oracle_text)
                full_tokens = prompt_tokens + oracle_tokens
                full_logprobs = sampling_client.compute_logprobs(types.ModelInput.from_ints(full_tokens)).result(timeout=900)
                # logprob for generated token at full index len(prompt_tokens) onward
                oracle_lps = [float(x or 0.0) for x in full_logprobs[len(prompt_tokens): len(prompt_tokens) + len(oracle_tokens)]]
                response_tokens.append(oracle_tokens)
                response_logprobs.append(oracle_lps)
                response_texts.append(oracle_text)

            raw_line = json.loads((ROOT / "data" / "fastapply_tiny_raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
            group_datums, group_rollouts = build_grpo_datums_for_group(
                prompt_id=rec["id"],
                prompt_tokens=prompt_tokens,
                response_token_groups=response_tokens,
                response_logprob_groups=response_logprobs,
                response_texts=response_texts,
                expected_source=raw_line["new_source"],
                language=rec["language"],
            )
            datums.extend(group_datums)
            rollout_json.extend([r.to_json() for r in group_rollouts])

        report["num_grpo_datums"] = len(datums)
        report["rollouts"] = rollout_json
        report["token_lengths"] = [len(d.model_input.to_ints()) for d in datums]
        print(f"[{now_iso()}] train GRPO datums={len(datums)}", flush=True)
        fb_result = training_client.forward_backward(datums, loss_fn="importance_sampling").result(timeout=1500)
        optim_result = training_client.optim_step(types.AdamParams(learning_rate=args.learning_rate)).result(timeout=1500)
        report["forward_backward_metrics"] = getattr(fb_result, "metrics", None)
        report["optim_step_result"] = str(optim_result)[:1000]
        report["step_completed_at"] = now_iso()

        final_sampler = training_client.save_weights_and_get_sampling_client(name=RUN_NAME)
        check = final_sampler.sample(
            prompt=types.ModelInput.from_ints(tokenizer.encode(records[0]["prompt"])),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.2, seed=23),
        ).result(timeout=900)
        report["saved_run_name"] = RUN_NAME
        report["post_grpo_sample"] = [tokenizer.decode(seq.tokens) for seq in check.sequences]
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        print(f"[{now_iso()}] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        report["ended_at"] = now_iso()
        report["duration_sec"] = round(time.time() - started, 2)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
