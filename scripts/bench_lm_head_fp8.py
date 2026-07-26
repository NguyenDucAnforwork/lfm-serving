#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm import _custom_ops as ops


def time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-safetensors",
        default=".hf_home/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9/model.safetensors",
    )
    ap.add_argument("--out-dir", default="results/lm_head_fp8_bench")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(args.model_safetensors, framework="pt", device="cpu") as f:
        weight_cpu = f.get_tensor("model.embed_tokens.weight")

    weight = weight_cpu.to(device="cuda", dtype=torch.bfloat16).contiguous()
    qweight, weight_scale = ops.scaled_fp8_quant(weight, scale=None)
    torch.cuda.synchronize()

    rows = []
    for batch in [1, 2, 4, 8]:
        torch.manual_seed(1234 + batch)
        x = torch.randn(batch, weight.shape[1], device="cuda", dtype=torch.bfloat16)
        x_q, x_scale = ops.scaled_fp8_quant(x.contiguous(), scale=None)

        y_bf16 = F.linear(x, weight)
        y_fp8 = torch._scaled_mm(
            x_q,
            qweight.t(),
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()

        bf16_ms = time_cuda(lambda: F.linear(x, weight), args.warmup, args.iters)

        def fp8_run():
            xq, xs = ops.scaled_fp8_quant(x.contiguous(), scale=None)
            return torch._scaled_mm(
                xq,
                qweight.t(),
                scale_a=xs,
                scale_b=weight_scale,
                out_dtype=torch.bfloat16,
            )

        fp8_ms = time_cuda(fp8_run, args.warmup, args.iters)
        max_abs = (y_bf16.float() - y_fp8.float()).abs().max().item()
        mean_abs = (y_bf16.float() - y_fp8.float()).abs().mean().item()
        rows.append(
            {
                "batch": batch,
                "bf16_ms": bf16_ms,
                "fp8_dynamic_ms": fp8_ms,
                "delta_ms": fp8_ms - bf16_ms,
                "speedup": bf16_ms / fp8_ms if fp8_ms else None,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
            }
        )
        print(rows[-1], flush=True)

    with (out_dir / "lm_head_fp8_bench.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
