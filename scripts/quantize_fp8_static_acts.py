#!/usr/bin/env python3
"""Quantize LFM2.5 to offline W8A8 FP8 with calibrated static activations.

This candidate targets the dense attention and MLP projections only:

* attention q/k/v/out: FP8 weights + static tensor input scale
* feed_forward w1/w2/w3: FP8 weights + static tensor input scale
* ShortConv projections/depthwise conv: left BF16
* tied embedding/lm_head: left BF16

The calibration dataset is synthetic but shaped from trace_grading_spec_v2:
system prefix first, growing multi-turn conversation, 2150-4400 token prompt
lengths, and assistant text included so teacher-forcing calibration sees both
prompt-like and decode-like positions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


FP8_STATIC_TARGETS = [
    r"re:model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$",
    r"re:model\.layers\.\d+\.feed_forward\.(w1|w2|w3)$",
]


def build_recipe(weight_strategy: str):
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )
    from compressed_tensors.config import CompressionFormat
    from llmcompressor.modifiers.quantization import QuantizationModifier

    strategy = {
        "channel": QuantizationStrategy.CHANNEL,
        "tensor": QuantizationStrategy.TENSOR,
    }[weight_strategy]

    scheme = QuantizationScheme(
        targets=FP8_STATIC_TARGETS,
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=strategy,
            symmetric=True,
            dynamic=False,
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=False,
            observer="static_minmax",
        ),
        format=CompressionFormat.float_quantized,
    )

    return [
        QuantizationModifier(
            config_groups={"group_0": scheme},
        )
    ]


def load_trace_lengths(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return [(2150 + (i % 6) * 450, i % 6) for i in range(420)]

    lengths: list[tuple[int, int]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            lengths.append((int(row["in_tokens_est"]), int(row["turn_idx"])))
    return lengths


def _make_turn(topic: str, turn: int, user_words: int, assistant_words: int) -> str:
    user = (
        f"User turn {turn} about {topic}. "
        "Please keep the answer precise, compare options, mention constraints, "
        "and preserve previous requirements. "
    )
    assistant = (
        f"Assistant turn {turn}: for {topic}, the practical answer is to isolate "
        "the variable, run a controlled measurement, and record accuracy, latency, "
        "memory, and failure mode. "
    )
    user += " ".join(
        f"requirement_{turn}_{i % 37}" for i in range(max(0, user_words))
    )
    assistant += " ".join(
        f"observation_{turn}_{i % 41}" for i in range(max(0, assistant_words))
    )
    return user + "\n" + assistant + "\n"


def build_calibration_dataset(
    tokenizer,
    trace_path: Path,
    num_samples: int,
    max_seq_len: int,
    seed: int,
):
    from datasets import Dataset

    rng = random.Random(seed)
    trace = load_trace_lengths(trace_path)
    topics = [
        "GPU serving latency",
        "quantized language model deployment",
        "multi turn customer support",
        "Python debugging workflow",
        "cloud infrastructure incident",
        "structured data extraction",
        "Vietnamese English translation",
        "technical benchmark analysis",
    ]

    rows = []
    for idx in range(num_samples):
        target_tokens, turn_idx = trace[idx % len(trace)]
        target_tokens = min(target_tokens, max_seq_len)
        topic = topics[idx % len(topics)]

        system_words = 720 + (idx % 11) * 9
        system = (
            "System: You are a concise technical assistant. Follow policy, maintain "
            "context, avoid unnecessary wording, and answer with actionable detail. "
        )
        system += " ".join(f"policy_{i % 53}" for i in range(system_words)) + "\n"

        remaining_words = max(64, int(target_tokens * 0.72) - system_words)
        turns = max(1, turn_idx + 1)
        pieces = [system]
        for t in range(turns):
            share = remaining_words / turns
            jitter = rng.randint(-12, 12)
            user_words = max(16, math.floor(share * 0.56) + jitter)
            assistant_words = max(16, math.floor(share * 0.44) - jitter)
            pieces.append(_make_turn(topic, t, user_words, assistant_words))

        text = "".join(pieces)
        encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_seq_len,
        )
        rows.append(encoded)

    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default=".hf_home/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9",
    )
    ap.add_argument(
        "--output",
        default="artifacts/lfm2-fp8-static-acts-attn-mlp-bf16conv-lmhead-v1",
    )
    ap.add_argument("--trace", default="trace_grading_spec_v2.jsonl")
    ap.add_argument("--num-calibration-samples", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=4400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-strategy", choices=["channel", "tensor"], default="channel")
    ap.add_argument("--gpu-max-memory", default="14GiB")
    args = ap.parse_args()

    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer/model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_max_memory, "cpu": "64GiB"},
    )

    print(
        "Building trace-shaped calibration dataset "
        f"({args.num_calibration_samples} samples, max_seq_length={args.max_seq_length}) ..."
    )
    dataset = build_calibration_dataset(
        tokenizer=tokenizer,
        trace_path=Path(args.trace),
        num_samples=args.num_calibration_samples,
        max_seq_len=args.max_seq_length,
        seed=args.seed,
    )

    recipe = build_recipe(args.weight_strategy)
    print(
        "Running offline FP8 static activation quantization "
        f"(weight_strategy={args.weight_strategy}) ..."
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving quantized checkpoint to {out_dir} ...")
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
