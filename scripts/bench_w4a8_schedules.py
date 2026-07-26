#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    convert_bf16_scales_to_fp8,
    convert_packed_uint4b8_to_signed_int4_inplace,
)


SCHEDULES = [
    None,
    "128x16_1x1x1",
    "128x16_1x1x1_TmaMI__TmaCoop_streamK",
    "128x32_1x1x1",
    "128x32_2x1x1_TmaMI__TmaCoop_streamK",
    "128x64_1x1x1",
    "128x64_2x1x1_TmaMI__TmaCoop_streamK",
    "128x128_1x1x1",
    "128x128_2x1x1_TmaMI__TmaCoop_streamK",
    "128x256_1x1x1",
    "128x256_2x1x1",
    "128x256_2x1x1_TmaMI__TmaCoop_streamK",
    "256x16_1x1x1",
    "256x16_1x1x1_TmaMI__TmaCoop_streamK",
    "256x32_1x1x1",
    "256x32_2x1x1_TmaMI__TmaCoop_streamK",
    "256x64_1x1x1",
    "256x64_2x1x1_TmaMI__TmaCoop_streamK",
    "256x128_1x1x1",
    "256x128_2x1x1_TmaMI__TmaCoop_streamK",
]


def quant_fp8_dynamic_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)


def load_w4(path: Path, prefix: str):
    with safe_open(path, framework="pt", device="cpu") as f:
        w = f.get_tensor(f"{prefix}.weight_packed").cuda()
        s = f.get_tensor(f"{prefix}.weight_scale").cuda().contiguous()

    fp8_scales, chan_scales = convert_bf16_scales_to_fp8(
        quant_fp8_dynamic_per_token, s
    )

    # Match CutlassW4A8LinearKernel.process_weights_after_loading().
    convert_packed_uint4b8_to_signed_int4_inplace(w)
    torch.cuda.synchronize()
    w = w.permute(1, 0).contiguous()
    w = ops.cutlass_encode_and_reorder_int4b(w.t().contiguous().t())

    fp8_scales = fp8_scales.permute(1, 0).contiguous().to(torch.float8_e4m3fn)
    fp8_scales = ops.cutlass_pack_scale_fp8(fp8_scales)
    return w, fp8_scales, chan_scales


def make_fp8_weight(k: int, n: int):
    weight_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02
    scale = weight_bf16.abs().amax(dim=0, keepdim=True).float().clamp(min=1e-6) / 448.0
    weight_fp8 = (weight_bf16.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return weight_fp8.contiguous(), scale.squeeze(0).contiguous()


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


def bench_shape(name: str, w4_prefix: str, k: int, n: int, ms: list[int], args):
    w4, w4_scale, w4_chan_scale = load_w4(args.artifact / "model.safetensors", w4_prefix)
    fp8_w, fp8_ws = make_fp8_weight(k, n)
    rows = []
    for m in ms:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

        act_fp8, act_scale = quant_fp8_dynamic_per_token(x)
        bf16_w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02
        bf16_ms = time_cuda(lambda: x @ bf16_w, args.warmup, args.iters)
        try:
            fp8_raw_ms = time_cuda(
                lambda: ops.cutlass_scaled_mm(
                    act_fp8,
                    fp8_w,
                    out_dtype=torch.bfloat16,
                    scale_a=act_scale,
                    scale_b=fp8_ws,
                    bias=None,
                ),
                args.warmup,
                args.iters,
            )
            fp8_total_ms = time_cuda(
                lambda: (
                    lambda q: ops.cutlass_scaled_mm(
                        q[0],
                        fp8_w,
                        out_dtype=torch.bfloat16,
                        scale_a=q[1],
                        scale_b=fp8_ws,
                        bias=None,
                    )
                )(quant_fp8_dynamic_per_token(x)),
                args.warmup,
                args.iters,
            )
            fp8_error = ""
        except Exception as exc:
            fp8_raw_ms = None
            fp8_total_ms = None
            fp8_error = str(exc).splitlines()[0][:240]

        best = None
        for schedule in SCHEDULES:
            try:
                raw_ms = time_cuda(
                    lambda schedule=schedule: ops.cutlass_w4a8_mm(
                        a=act_fp8,
                        b_q=w4,
                        b_group_scales=w4_scale,
                        b_group_size=128,
                        b_channel_scales=w4_chan_scale,
                        a_token_scales=act_scale,
                        maybe_schedule=schedule,
                    ),
                    args.warmup,
                    args.iters,
                )
                total_ms = time_cuda(
                    lambda schedule=schedule: (
                        lambda q: ops.cutlass_w4a8_mm(
                            a=q[0],
                            b_q=w4,
                            b_group_scales=w4_scale,
                            b_group_size=128,
                            b_channel_scales=w4_chan_scale,
                            a_token_scales=q[1],
                            maybe_schedule=schedule,
                        )
                    )(quant_fp8_dynamic_per_token(x)),
                    args.warmup,
                    args.iters,
                )
                status = "ok"
                err = ""
                if best is None or total_ms < best["w4_total_ms"]:
                    best = {
                        "schedule": schedule or "default",
                        "w4_raw_ms": raw_ms,
                        "w4_total_ms": total_ms,
                    }
            except Exception as exc:
                raw_ms = None
                total_ms = None
                status = "fail"
                err = str(exc).splitlines()[0][:240]

            rows.append(
                {
                    "shape": name,
                    "m": m,
                    "k": k,
                    "n": n,
                    "schedule": schedule or "default",
                    "status": status,
                    "w4_raw_ms": raw_ms,
                    "w4_total_ms": total_ms,
                    "fp8_raw_ms": fp8_raw_ms,
                    "fp8_total_ms": fp8_total_ms,
                    "bf16_ms": bf16_ms,
                    "error": err,
                    "fp8_error": fp8_error,
                }
            )

        print(
            f"{name} M={m}: bf16={bf16_ms:.4f}ms "
            f"best_w4={best['w4_total_ms']:.4f}ms {best['schedule']}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/lfm2-w4afp8-fused-w13-vllm-packed-lmhead-bf16-v2"))
    parser.add_argument("--out", type=Path, default=Path("results/w4a8_schedule_bench"))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--ms", default="1,2,4,8,16")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ms = [int(x) for x in args.ms.split(",") if x]
    all_rows = []
    all_rows += bench_shape(
        "w13",
        "model.layers.0.feed_forward.w13",
        2048,
        16384,
        ms,
        args,
    )
    all_rows += bench_shape(
        "w2",
        "model.layers.0.feed_forward.w2",
        8192,
        2048,
        ms,
        args,
    )

    csv_path = args.out / "w4a8_schedule_bench.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {}
    for shape in ["w13", "w2"]:
        summary[shape] = {}
        for m in ms:
            rows = [r for r in all_rows if r["shape"] == shape and r["m"] == m and r["status"] == "ok"]
            best = min(rows, key=lambda r: r["w4_total_ms"])
            summary[shape][str(m)] = best
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
