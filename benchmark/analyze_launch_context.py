#!/usr/bin/env python3
"""Analyze CUDA runtime launch/copy context in torch profiler traces.

This complements analyze_profile_trace.py. It answers the narrower question:
which CPU ops enclose cudaLaunchKernel/cudaMemcpyAsync/cudaGraphLaunch calls,
and how many such calls occur per profiler step.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUNTIME_NAMES = {
    "cudaLaunchKernel",
    "cudaGraphLaunch",
    "cudaMemcpyAsync",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaStreamIsCapturing",
}


def load_events(profile_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    rank_traces = sorted(profile_dir.glob("rank0*.pt.trace.json.gz"))
    if not rank_traces:
        raise FileNotFoundError(f"no rank0 torch trace found in {profile_dir}")
    trace = rank_traces[-1]
    with gzip.open(trace, "rt") as f:
        data = json.load(f)
    events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]
    return trace, events


def enclosing_op(ts: float, containers: list[dict[str, Any]]) -> str:
    matches = [
        c for c in containers if c.get("ts", 0) <= ts <= c.get("ts", 0) + c.get("dur", 0)
    ]
    cpu_matches = [c for c in matches if c.get("cat") == "cpu_op"]
    if cpu_matches:
        return min(cpu_matches, key=lambda e: e.get("dur", 10**18)).get("name", "")
    if matches:
        return min(matches, key=lambda e: e.get("dur", 10**18)).get("name", "")
    return "<no enclosing op>"


def analyze(profile_dir: Path) -> dict[str, Any]:
    trace, events = load_events(profile_dir)
    containers = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")]
    steps = sorted(
        [
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and e.get("name", "").startswith("ProfilerStep#")
        ],
        key=lambda e: e["ts"],
    )

    runtime_counts: Counter[str] = Counter()
    runtime_durations: Counter[str] = Counter()
    runtime_cpu_counts: dict[str, Counter[str]] = defaultdict(Counter)
    runtime_cpu_durations: dict[str, Counter[str]] = defaultdict(Counter)
    kernel_counts: Counter[str] = Counter()
    memcpy_counts: Counter[str] = Counter()
    cpu_op_counts: Counter[str] = Counter()

    for e in events:
        cat = e.get("cat")
        name = e.get("name", "")
        dur_ms = e.get("dur", 0) / 1000.0
        if cat == "cuda_runtime":
            runtime_counts[name] += 1
            runtime_durations[name] += dur_ms
            if name in RUNTIME_NAMES:
                key = enclosing_op(e.get("ts", -1), containers)
                runtime_cpu_counts[name][key] += 1
                runtime_cpu_durations[name][key] += dur_ms
        elif cat == "kernel":
            kernel_counts[name] += 1
        elif cat == "gpu_memcpy":
            memcpy_counts[name] += 1
        elif cat == "cpu_op":
            cpu_op_counts[name] += 1

    per_step = []
    for step in steps:
        lo = step["ts"]
        hi = lo + step.get("dur", 0)
        ctr: Counter[str] = Counter()
        kernels = 0
        mem: Counter[str] = Counter()
        for e in events:
            ts = e.get("ts", -1)
            if not (lo <= ts <= hi):
                continue
            if e.get("cat") == "cuda_runtime" and e.get("name") in RUNTIME_NAMES:
                ctr[e["name"]] += 1
            elif e.get("cat") == "kernel":
                kernels += 1
            elif e.get("cat") == "gpu_memcpy":
                mem[e.get("name", "")] += 1
        per_step.append(
            {
                "step": step["name"],
                "duration_ms": step.get("dur", 0) / 1000.0,
                "runtime_counts": dict(sorted(ctr.items())),
                "kernel_events": kernels,
                "memcpy_counts": mem.most_common(),
            }
        )

    step_durations = [s["duration_ms"] for s in per_step]
    return {
        "trace": str(trace),
        "n_profiler_steps": len(per_step),
        "step_duration_ms": {
            "mean": statistics.mean(step_durations) if step_durations else 0,
            "p50": statistics.median(step_durations) if step_durations else 0,
            "max": max(step_durations) if step_durations else 0,
        },
        "cuda_runtime_counts": runtime_counts.most_common(30),
        "cuda_runtime_duration_ms": [
            (k, runtime_durations[k]) for k, _ in runtime_counts.most_common(30)
        ],
        "runtime_cpu_context": {
            name: [
                {
                    "op": op,
                    "count": count,
                    "duration_ms": runtime_cpu_durations[name][op],
                }
                for op, count in runtime_cpu_counts[name].most_common(30)
            ]
            for name in sorted(runtime_cpu_counts)
        },
        "top_kernel_counts": kernel_counts.most_common(30),
        "top_memcpy_counts": memcpy_counts.most_common(),
        "top_cpu_op_counts": cpu_op_counts.most_common(30),
        "per_step": per_step,
    }


def write_text(report: dict[str, Any], out: Path) -> None:
    with open(out, "w") as f:
        f.write(f"Trace: {report['trace']}\n")
        f.write(f"Profiler steps: {report['n_profiler_steps']}\n")
        s = report["step_duration_ms"]
        f.write(
            "Step duration ms: "
            f"mean={s['mean']:.3f} p50={s['p50']:.3f} max={s['max']:.3f}\n\n"
        )
        f.write("CUDA runtime counts/durations:\n")
        durations = dict(report["cuda_runtime_duration_ms"])
        for name, count in report["cuda_runtime_counts"]:
            f.write(f"  {count:4d} {durations[name]:8.3f} ms  {name}\n")
        f.write("\nRuntime CPU context:\n")
        for name, rows in report["runtime_cpu_context"].items():
            f.write(f"  {name}:\n")
            for row in rows[:15]:
                f.write(
                    f"    {row['count']:4d} {row['duration_ms']:8.3f} ms  {row['op']}\n"
                )
        f.write("\nTop kernel counts:\n")
        for name, count in report["top_kernel_counts"]:
            f.write(f"  {count:4d} {name[:180]}\n")
        f.write("\nPer-step runtime counts:\n")
        for row in report["per_step"]:
            f.write(
                f"  {row['step']} dur={row['duration_ms']:.3f}ms "
                f"runtime={row['runtime_counts']} kernels={row['kernel_events']} "
                f"memcpy={row['memcpy_counts']}\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--text-out", type=Path)
    args = ap.parse_args()

    report = analyze(args.profile_dir)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
    if args.text_out:
        write_text(report, args.text_out)
    if not args.json_out and not args.text_out:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
