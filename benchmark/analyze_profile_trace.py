#!/usr/bin/env python3
"""Summarize vLLM/Torch Chrome traces into decode-cost buckets.

This is a heuristic attribution tool for the requested decode-cost work. It
accepts one or more Chrome trace JSON/JSON.GZ files or directories containing
them, sums event durations, and groups ops/kernels into buckets relevant to
LFM2 serving:

  gemm, fp8_scaling, gated_conv_state, attention, launch_overhead, other

The output is intentionally transparent: the JSON includes top events per
bucket so suspicious matches can be inspected before making optimization calls.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BUCKETS = (
    "gemm",
    "fp8_scaling",
    "gated_conv_state",
    "attention",
    "launch_overhead",
    "other",
)


def iter_trace_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
            files.extend(sorted(p.rglob("*.json.gz")))
        elif p.is_file():
            files.append(p)
    # Keep deterministic order and avoid double-counting symlink/explicit dupes.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        events = []
    return [e for e in events if isinstance(e, dict)]


def event_duration_us(event: dict[str, Any]) -> float:
    dur = event.get("dur")
    if isinstance(dur, (int, float)):
        return float(dur)
    # Complete events normally have dur. Instant/counter events do not.
    return 0.0


def include_event(event: dict[str, Any]) -> bool:
    """Keep non-overlapping CUDA/kernel/runtime events.

    Torch profiler Chrome traces contain large wrapper ranges such as
    "PyTorch Profiler", "ProfilerStep", and execute_context annotations. Those
    overlap the actual CUDA kernels and make bucket percentages meaningless if
    summed together. For decode-cost attribution we want the concrete kernel /
    CUDA runtime work.
    """

    cat = str(event.get("cat", ""))
    name = str(event.get("name", ""))
    if cat in {"Trace", "user_annotation", "gpu_user_annotation", "cpu_op"}:
        return False
    if name.startswith("ProfilerStep") or name.startswith("PyTorch Profiler"):
        return False
    return cat in {"kernel", "cuda_runtime", "cuda_driver", "gpu_memcpy", "overhead"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def classify(name: str, cat: str = "") -> str:
    s = norm(f"{name} {cat}")

    if re.search(
        r"(attention|flash.?attn|flash_fwd|paged.?attention|unified.?attention|"
        r"flashinfer|xformers|kv.?cache|reshape_and_cache|rotary|rope)",
        s,
    ):
        return "attention"

    # Order matters. Fused W4 kernels often include quantization-ish words but
    # should count as GEMM if the actual work is Marlin/Machete/CUTLASS matmul.
    if re.search(
        r"(machete|marlin|cutlass|cublas|gemm|grouped_gemm|scaled_mm|"
        r"matmul|matrixmultiply|aten::mm|aten::bmm|aten::addmm|"
        r"aten::linear|triton.*mm|wna16|w4a16)",
        s,
    ):
        return "gemm"

    if re.search(
        r"(float8|fp8|amax|dynamic.*scale|scale.*dynamic|quantiz|dequant|"
        r"requant|scaled.*cast|cast.*scaled|reduce_amax|amax_reduce|"
        r"torch._scaled|_scaled_mm_pre)",
        s,
    ):
        return "fp8_scaling"

    if re.search(
        r"(short.?conv|causal.?conv|gated.?conv|depthwise.?conv|conv1d|"
        r"state.?update|cache.?update|ssm|mamba|mixer|scan|selective)",
        s,
    ):
        return "gated_conv_state"

    if re.search(
        r"(cudalaunch|cuda_launch|cuda kernel launch|cudaapis|cuda runtime|"
        r"cudamemcpy|cudaMemcpy|cudadevicesynchronize|cudaStreamSynchronize|"
        r"cudaevent|cuda graph|cudaGraphLaunch|profilerstep|python)",
        f"{name} {cat}",
        re.IGNORECASE,
    ):
        return "launch_overhead"

    return "other"


def summarize(files: list[Path], top_n: int) -> dict[str, Any]:
    bucket_us = Counter({b: 0.0 for b in BUCKETS})
    bucket_count = Counter({b: 0 for b in BUCKETS})
    top: dict[str, Counter[str]] = {b: Counter() for b in BUCKETS}
    event_total = 0
    dur_total_us = 0.0

    for path in files:
        for event in load_trace_events(path):
            if not include_event(event):
                continue
            dur_us = event_duration_us(event)
            if dur_us <= 0:
                continue
            name = str(event.get("name", ""))
            cat = str(event.get("cat", ""))
            bucket = classify(name, cat)
            bucket_us[bucket] += dur_us
            bucket_count[bucket] += 1
            top[bucket][name] += dur_us
            event_total += 1
            dur_total_us += dur_us

    buckets: dict[str, Any] = {}
    for bucket in BUCKETS:
        us = bucket_us[bucket]
        buckets[bucket] = {
            "duration_ms": us / 1000.0,
            "pct": (100.0 * us / dur_total_us) if dur_total_us else 0.0,
            "event_count": bucket_count[bucket],
            "top_events": [
                {"name": name, "duration_ms": value / 1000.0}
                for name, value in top[bucket].most_common(top_n)
            ],
        }

    return {
        "files": [str(p) for p in files],
        "event_count": event_total,
        "total_duration_ms": dur_total_us / 1000.0,
        "buckets": buckets,
    }


def print_table(summary: dict[str, Any]) -> None:
    print(f"Trace files: {len(summary['files'])}")
    print(f"Timed events: {summary['event_count']}")
    print(f"Summed event duration: {summary['total_duration_ms']:.3f} ms")
    print()
    print(f"{'bucket':<20} {'duration_ms':>14} {'pct':>8} {'events':>10}")
    print("-" * 56)
    for bucket in BUCKETS:
        data = summary["buckets"][bucket]
        print(
            f"{bucket:<20} {data['duration_ms']:>14.3f} "
            f"{data['pct']:>7.2f}% {data['event_count']:>10}"
        )
    print()
    for bucket in BUCKETS:
        events = summary["buckets"][bucket]["top_events"]
        if not events:
            continue
        print(f"Top {bucket}:")
        for event in events[:5]:
            print(f"  {event['duration_ms']:10.3f} ms  {event['name']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Chrome trace JSON/JSON.GZ files or directories")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--json-out", default=None, help="Optional path for full JSON summary")
    args = ap.parse_args()

    files = iter_trace_files(args.paths)
    if not files:
        raise SystemExit("No trace JSON/JSON.GZ files found")

    summary = summarize(files, args.top_n)
    print_table(summary)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
