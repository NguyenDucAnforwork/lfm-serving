#!/usr/bin/env python3
"""Export LFM2.5 as W4A16 GPTQ in compressed-tensors format.

Run from the separate quantization env:

  source .venv-quant/bin/activate
  python scripts/quantize_w4a16_gptq.py \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --output-dir artifacts/lfm2-w4a16-gptq-g128

Serve with:

  MODEL=artifacts/lfm2-w4a16-gptq-g128 QUANTIZATION=compressed-tensors \
  LINEAR_BACKEND=machete bash scripts/start_server.sh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_calibration_dataset(n: int) -> Dataset:
    rows = []
    for i in range(n):
        body = " ".join(
            f"section_{i % 17}_{j}: latency throughput memory quantization cache"
            for j in range(220)
        )
        rows.append(
            {
                "text": (
                    "You are answering a technical reasoning question. "
                    "Read the context carefully, preserve exact constraints, "
                    f"and answer concisely.\n{body}\nQuestion {i}: identify "
                    "the bottleneck and provide the final answer."
                )
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--recipe", default="recipes/w4a16_gptq_group128.yaml")
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
        dataset=build_calibration_dataset(args.num_calibration_samples),
        text_column="text",
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        pad_to_max_length=False,
        save_compressed=True,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
