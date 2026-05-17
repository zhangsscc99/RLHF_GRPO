#!/usr/bin/env python3
"""MinT FastApply SFT smoke test.

Uses a tiny Apply dataset that mirrors the photographed FastApply pipeline:
source + patch -> <updated>new source</updated>.
The API key is read only from MINT_API_KEY and is never printed or saved.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "fastapply_mint_training_report.json"
BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
RUN_NAME = os.environ.get("MINT_RUN_NAME", "fastapply-qwen4b-tiny-2026-05-17")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_sft_records() -> list[dict[str, Any]]:
    if not SFT_PATH.exists():
        build_dataset()
    return [json.loads(line) for line in SFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def datum_from_record(record: dict[str, Any], tokenizer) -> types.Datum:
    prompt_ids = tokenizer.encode(record["prompt"])
    response_ids = tokenizer.encode(record["response"])
    all_tokens = prompt_ids + response_ids
    all_weights = [0.0] * len(prompt_ids) + [1.0] * len(response_ids)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": all_tokens[1:],
            "weights": all_weights[1:],
        },
    )


def main() -> int:
    started = time.time()
    records = load_sft_records()
    report: dict[str, Any] = {
        "started_at": now_iso(),
        "pipeline": "FastApply source+patch -> updated source",
        "base_url": os.environ.get("MINT_BASE_URL", "https://mint.macaron.xin/"),
        "requested_model_note": "Using the MinT-supported Qwen3 4B model: Qwen/Qwen3-4B-Instruct-2507.",
        "base_model": BASE_MODEL,
        "algorithm": "SFT / cross_entropy",
        "lora_rank": 16,
        "dataset_path": str(SFT_PATH),
        "dataset_examples": len(records),
        "record_ids": [r["id"] for r in records],
        "steps_requested": 1,
        "api_key": "provided via MINT_API_KEY (redacted)",
    }
    try:
        if not os.environ.get("MINT_API_KEY"):
            raise RuntimeError("MINT_API_KEY is not set")

        print(f"[{now_iso()}] creating MinT client model={BASE_MODEL}", flush=True)
        service_client = mint.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=16,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"purpose": "fastapply-pipeline-smoke", "date": "2026-05-17"},
        )
        tokenizer = training_client.get_tokenizer()
        data = [datum_from_record(record, tokenizer) for record in records]
        report["token_lengths"] = [len(d.model_input.to_ints()) for d in data]

        print(f"[{now_iso()}] running one SFT step", flush=True)
        fb_future = training_client.forward_backward(data, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=5e-5))
        fb_result = fb_future.result(timeout=1200)
        optim_result = optim_future.result(timeout=1200)
        report["forward_backward_metrics"] = getattr(fb_result, "metrics", None)
        report["optim_step_result"] = str(optim_result)[:1000]
        report["step_completed_at"] = now_iso()

        print(f"[{now_iso()}] saving weights and sampling", flush=True)
        sampling_client = training_client.save_weights_and_get_sampling_client(name=RUN_NAME)
        sample_prompt = records[0]["prompt"]
        sample_future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokenizer.encode(sample_prompt)),
            sampling_params=types.SamplingParams(max_tokens=180, temperature=0.2, seed=17),
            num_samples=1,
        )
        sample_response = sample_future.result(timeout=600)
        report["saved_run_name"] = RUN_NAME
        report["sample_record_id"] = records[0]["id"]
        report["sample_outputs"] = [tokenizer.decode(seq.tokens) for seq in sample_response.sequences]
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
