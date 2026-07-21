#!/usr/bin/env python3
"""Export LFM2.5 as W4A16 GPTQ in compressed-tensors format.

Run from the separate quantization env:

  source .venv-quant/bin/activate
  python scripts/quantize_w4a16_gptq.py \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --output-dir artifacts/lfm2-w4a16-gptq-g128-no-conv

Serve with:

  MODEL=artifacts/lfm2-w4a16-gptq-g128-no-conv QUANTIZATION=compressed-tensors \
  LINEAR_BACKEND=machete bash scripts/start_server.sh

The default recipe keeps LFM2 short-conv projections unquantized because current
vLLM constructs ShortConv without quant_config and cannot load packed W4 tensors
for those modules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_calibration_dataset(tokenizer, n: int, max_seq_length: int) -> Dataset:
    rows = []
    for i in range(n):
        # Chat-shaped calibration that resembles the competition trace: long
        # technical context, explicit constraints, and concise assistant-style
        # answers. Keep it deterministic and local; no external dataset needed.
        target_words = max(360, min(1200, max_seq_length // 2))
        body = " ".join(
            (
                f"section_{i % 31}_{j}: latency throughput memory quantization "
                "cache scheduler retry context attention bandwidth answer-format"
            )
            for j in range(target_words // 10)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful technical assistant. Preserve constraints, "
                    "reason over long context, and answer coherently."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{body}\n\nQuestion {i}: identify the main "
                    "bottleneck, explain the risk, and provide the final answer "
                    "in a compact paragraph."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The main bottleneck is decode-side latency from repeated "
                    "kernel work and memory movement. The safest answer is to "
                    "optimize the measured hot path while preserving accuracy."
                ),
            },
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--recipe", default="recipes/w4a16_gptq_group128_no_conv.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-calibration-samples", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=args.recipe,
        dataset=build_calibration_dataset(
            tokenizer, args.num_calibration_samples, args.max_seq_length
        ),
        text_column="text",
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        pad_to_max_length=False,
        save_compressed=True,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
