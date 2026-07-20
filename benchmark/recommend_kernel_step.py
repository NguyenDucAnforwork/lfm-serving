#!/usr/bin/env python3
"""Recommend the next kernel-level step from profile bucket evidence.

This does not claim a benchmark result. It converts
benchmark/analyze_profile_trace.py JSON into one concrete next action, matching
the project rule that static FP8 or kernel fusion should only be attempted when
profiling identifies the corresponding hotspot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def bucket_pct(profile: dict[str, Any], name: str) -> float:
    return float(((profile.get("buckets") or {}).get(name) or {}).get("pct") or 0.0)


def top_events(profile: dict[str, Any], name: str, n: int = 5) -> list[str]:
    bucket = ((profile.get("buckets") or {}).get(name) or {})
    return [str(e.get("name")) for e in (bucket.get("top_events") or [])[:n]]


def recommend(profile: dict[str, Any]) -> dict[str, Any]:
    pcts = {name: bucket_pct(profile, name) for name in (
        "gemm",
        "fp8_scaling",
        "gated_conv_state",
        "attention",
        "launch_overhead",
        "other",
    )}

    # Thresholds are intentionally conservative. Below these levels the likely
    # gain is too small to justify adding accuracy/runtime risk.
    if pcts["fp8_scaling"] >= 10.0:
        step = (
            "Build an offline calibrated/static FP8 artifact or vLLM quant config "
            "that removes per-request/per-step dynamic amax/scale reductions; rerun "
            "the matched FP8 profile to confirm the fp8_scaling bucket shrinks before "
            "running accuracy gates."
        )
        area = "static_fp8"
    elif pcts["gated_conv_state"] >= 15.0:
        step = (
            "Prototype a fused LFM2 gated-conv/state-update decode kernel for the "
            "dominant short_conv/state_update ops, preserving CUDA graph capture; "
            "benchmark against the matched FP8 baseline before changing quantization."
        )
        area = "gated_conv_state_fusion"
    elif pcts["gemm"] >= 55.0:
        step = (
            "Prioritize fused W4A16 GPTQ/compressed-tensors with Machete on H200 "
            "and Marlin fallback, because the profile is GEMM-dominated and weight "
            "bandwidth reduction is the highest-ROI path."
        )
        area = "w4a16_gptq"
    elif pcts["launch_overhead"] >= 10.0:
        step = (
            "Investigate CUDA graph/launch fusion for decode step boundaries; do "
            "not disable cudagraphs, because prior eager testing showed launch/Python "
            "overhead is catastrophic under CPU limits."
        )
        area = "launch_overhead"
    elif pcts["attention"] >= 15.0:
        step = (
            "Inspect attention backend selection and KV-cache update kernels for the "
            "dominant attention events; only consider attention-specific kernel changes "
            "if the top events are decode attention rather than prefill artifacts."
        )
        area = "attention"
    else:
        step = (
            "No single bucket is dominant enough for a risky kernel change; run the "
            "W4A16 Machete/Marlin experiment first, then revisit top uncategorized "
            "events if W4 misses the decode target."
        )
        area = "w4a16_gptq"

    return {
        "bucket_percentages": pcts,
        "recommended_area": area,
        "recommended_next_step": step,
        "top_events": {
            name: top_events(profile, name)
            for name in pcts
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile_buckets_json")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    profile = json.loads(Path(args.profile_buckets_json).read_text())
    rec = recommend(profile)
    print(f"Recommended area: {rec['recommended_area']}")
    print(rec["recommended_next_step"])
    print("Bucket percentages:")
    for name, pct in rec["bucket_percentages"].items():
        print(f"  {name}: {pct:.2f}%")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
