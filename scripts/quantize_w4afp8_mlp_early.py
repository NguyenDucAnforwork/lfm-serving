#!/usr/bin/env python3
"""Quantize LFM2.5-1.2B-Instruct: W4A8-FP8 for early MLP layers (0-9),
FP8 for attention + late MLP (10-15) + lm_head, BF16 for ShortConv (conv.*).

Candidate Q4-A: probes whether a real Hopper-native low-bit path (int4
weight + FP8 activation, group_size=128, matching vLLM's
CompressedTensorsW4A8Fp8 scheme -- confirmed present in the installed
vLLM 0.22.1/0.25.1 by reading source directly) can push TBT below the
FP8-everywhere floor of 4ms observed across every H200 submission so far.

ShortConv layers are intentionally left unquantized: the ShortConv
FP8 quantization fix was tested and rejected on real H200 data (see
SUBMISSION_RESULTS.md "VERDICT (2026-07-25)") -- no reason to reintroduce
that variable here.

Usage:
    source .venv-quant/bin/activate
    python3 scripts/quantize_w4afp8_mlp_early.py \
        --model submission/model \
        --output artifacts/lfm2-w4afp8-mlp0-9-fp8-rest

Requires .venv-quant (llmcompressor, torch, transformers, datasets) --
see README "Reproduce from scratch" for install; NOT the main .venv used
for vLLM serving experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Early-MLP W4A8 target (layers 0-9 only -- the trailing \. after [0-9]
# means this regex does NOT match layers 10-15, since "10." doesn't match
# a single digit followed immediately by a literal dot).
EARLY_MLP_W4A8 = r"re:model\.layers\.[0-9]\.feed_forward\.(w1|w2|w3)$"

# Everything else that should be FP8: attention projections (all layers)
# and late-MLP (layers 10-15). lm_head included via explicit name below.
FP8_TARGETS = [
    r"re:model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$",
    r"re:model\.layers\.(1[0-5])\.feed_forward\.(w1|w2|w3)$",
    "lm_head",
]

# ShortConv (conv.*) is never touched by either modifier -- excluded from
# both target lists above, so it stays BF16 by default.


def build_recipe():
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    w4a8_early_mlp = GPTQModifier(
        targets=[EARLY_MLP_W4A8],
        scheme="W4AFP8",
        dampening_frac=0.1,
    )
    fp8_rest = QuantizationModifier(
        targets=FP8_TARGETS,
        scheme="FP8_DYNAMIC",
    )
    return [w4a8_early_mlp, fp8_rest]


def build_calibration_dataset(tokenizer, num_samples: int, max_seq_len: int):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(num_samples))

    def preprocess(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(
            text, padding=False, truncation=True, max_length=max_seq_len
        )

    return ds.map(preprocess, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="submission/model")
    ap.add_argument("--output", default="artifacts/lfm2-w4afp8-mlp0-9-fp8-rest")
    ap.add_argument("--num-calibration-samples", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="bfloat16", device_map="auto"
    )

    print(f"Building calibration dataset ({args.num_calibration_samples} samples, "
          f"max_seq_length={args.max_seq_length}) ...")
    dataset = build_calibration_dataset(tokenizer, args.num_calibration_samples, args.max_seq_length)

    recipe = build_recipe()

    print("Running oneshot quantization (GPTQ W4AFP8 on early MLP, "
          "FP8_DYNAMIC on attention + late MLP + lm_head, ShortConv left BF16) ...")
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
