#!/usr/bin/env python3
"""One-step MinT SFT smoke test.

Reads MINT_API_KEY from the environment. Does not print or persist the key.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import mint
from mint import types

BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RUN_NAME = os.environ.get("MINT_RUN_NAME", "codex-simple-sft-2026-05-17")
REPORT_PATH = Path(__file__).with_name("training_report.json")

EXAMPLES = [
    {
        "prompt": "请用一句话解释 MinT 是什么：",
        "response": "MinT 是把训练循环交给开发者、把 GPU 训练执行放到远程服务上的 LLM/RL 训练基础设施。",
    },
    {
        "prompt": "Q: 2 + 3 = ?\nA:",
        "response": " 5",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def datum_from_example(example: dict[str, str], tokenizer) -> types.Datum:
    prompt_ids = tokenizer.encode(example["prompt"])
    response_ids = tokenizer.encode(example["response"])
    all_tokens = prompt_ids + response_ids
    all_weights = [0.0] * len(prompt_ids) + [1.0] * len(response_ids)
    if len(all_tokens) < 2:
        raise ValueError("example produced fewer than 2 tokens")
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": all_tokens[1:],
            "weights": all_weights[1:],
        },
    )


def main() -> int:
    started = time.time()
    report: dict[str, object] = {
        "started_at": now_iso(),
        "base_url": os.environ.get("MINT_BASE_URL", "https://mint.macaron.xin"),
        "base_model": BASE_MODEL,
        "algorithm": "SFT / cross_entropy",
        "lora_rank": 16,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
        "dataset_examples": len(EXAMPLES),
        "steps_requested": 1,
        "api_key": "provided via MINT_API_KEY (redacted)",
    }

    if not os.environ.get("MINT_API_KEY"):
        report.update(status="failed", error="MINT_API_KEY is not set")
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    try:
        print(f"[{now_iso()}] creating MinT service client for {BASE_MODEL}", flush=True)
        service_client = mint.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=16,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"purpose": "codex-simple-sft-smoke", "date": "2026-05-17"},
        )
        report["training_client_created_at"] = now_iso()

        print(f"[{now_iso()}] loading tokenizer", flush=True)
        tokenizer = training_client.get_tokenizer()
        data = [datum_from_example(ex, tokenizer) for ex in EXAMPLES]
        report["token_lengths"] = [len(d.model_input.to_ints()) for d in data]

        print(f"[{now_iso()}] running one forward_backward + optim_step", flush=True)
        fb_future = training_client.forward_backward(data, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=5e-5))
        fb_result = fb_future.result(timeout=900)
        optim_result = optim_future.result(timeout=900)

        report["forward_backward_metrics"] = getattr(fb_result, "metrics", None)
        report["forward_backward_loss"] = getattr(fb_result, "loss", None)
        report["optim_step_result"] = str(optim_result)[:1000]
        report["step_completed_at"] = now_iso()

        print(f"[{now_iso()}] saving LoRA weights and doing tiny sample", flush=True)
        sampling_client = training_client.save_weights_and_get_sampling_client(name=RUN_NAME)
        prompt_ids = tokenizer.encode("Q: 2 + 3 = ?\nA:")
        sample_future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_ids),
            sampling_params=types.SamplingParams(max_tokens=12, temperature=0.2, seed=17),
            num_samples=1,
        )
        samples = sample_future.result(timeout=300)
        decoded = []
        for seq in samples.sequences:
            decoded.append(tokenizer.decode(seq.tokens))
        report["saved_run_name"] = RUN_NAME
        report["sample_prompt"] = "Q: 2 + 3 = ?\\nA:"
        report["sample_outputs"] = decoded
        report["status"] = "completed"
        print(f"[{now_iso()}] completed", flush=True)
    except Exception as exc:  # keep the report useful if the remote run fails
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        print(f"[{now_iso()}] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        report["ended_at"] = now_iso()
        report["duration_sec"] = round(time.time() - started, 2)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
