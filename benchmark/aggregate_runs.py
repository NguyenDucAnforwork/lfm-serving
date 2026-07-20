#!/usr/bin/env python3
"""Aggregate multiple benchmark summaries to assess candidate stability."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


FIELDS = (
    "ers_mean_pct",
    "tpot_ms_mean",
    "tpot_ms_p50",
    "tpot_ms_p95",
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "decode_tokens_per_s",
    "failed_requests",
    "successful_requests",
    "wall_time_s",
)


def load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "summary_ext.json"
    return json.loads(path.read_text())


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None, "stdev": None}
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="Run directories or summary_ext.json files")
    ap.add_argument("--baseline-tbt-p50-ms", type=float, default=4.0)
    ap.add_argument("--baseline-failed-requests", type=int, default=4)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    summaries = []
    for raw in args.runs:
        path = Path(raw)
        s = load_summary(path)
        s["_source"] = str(path)
        summaries.append(s)

    agg: dict[str, Any] = {
        "run_count": len(summaries),
        "sources": [s["_source"] for s in summaries],
        "baseline_tbt_p50_ms": args.baseline_tbt_p50_ms,
        "baseline_failed_requests": args.baseline_failed_requests,
        "fields": {},
    }
    for field in FIELDS:
        vals = [s.get(field) for s in summaries if isinstance(s.get(field), (int, float))]
        agg["fields"][field] = stats([float(v) for v in vals])

    tbt_p50_values = [s.get("tpot_ms_p50") for s in summaries if isinstance(s.get("tpot_ms_p50"), (int, float))]
    failed_values = [s.get("failed_requests") for s in summaries if isinstance(s.get("failed_requests"), (int, float))]
    max_tbt_p50 = max(tbt_p50_values) if tbt_p50_values else None
    max_failed = max(failed_values) if failed_values else None
    best_improvement = None
    worst_improvement = None
    if tbt_p50_values and args.baseline_tbt_p50_ms:
        improvements = [(args.baseline_tbt_p50_ms - v) / args.baseline_tbt_p50_ms for v in tbt_p50_values]
        best_improvement = max(improvements)
        worst_improvement = min(improvements)

    agg["gate_checks"] = {
        "has_at_least_3_runs": len(summaries) >= 3,
        "all_tbt_p50_le_3p5_ms": bool(tbt_p50_values) and max_tbt_p50 <= 3.5,
        "all_runs_ge_15pct_tbt_p50_improvement": worst_improvement is not None and worst_improvement >= 0.15,
        "no_extra_failures_vs_baseline": max_failed is not None and max_failed <= args.baseline_failed_requests,
        "max_tbt_p50_ms": max_tbt_p50,
        "best_tbt_p50_improvement_frac": best_improvement,
        "worst_tbt_p50_improvement_frac": worst_improvement,
        "max_failed_requests": max_failed,
    }

    print(f"Runs: {agg['run_count']}")
    for field in ("ers_mean_pct", "tpot_ms_p50", "tpot_ms_p95", "ttft_ms_p50", "failed_requests"):
        d = agg["fields"][field]
        print(f"{field}: min={d['min']} mean={d['mean']} max={d['max']} stdev={d['stdev']}")
    print("Gate checks:")
    for k, v in agg["gate_checks"].items():
        print(f"  {k}: {v}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
