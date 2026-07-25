#!/usr/bin/env python3
"""Compute the Efficiency/Response Score (ERS) from a replay_trace.py results file.

Scoring model (as specified for this competition):
  * TTFT: floor 10ms (score 1.0), ceiling 400ms (score 0.0)
  * TPOT: floor 1ms  (score 1.0), ceiling 10ms  (score 0.0)
  * equal weight between TTFT and TPOT (0.5 / 0.5)
  * gamma = 2 shaping exponent
  * errors, timeouts, or zero-token outputs score 0 for that request

The exact published formula for combining floor/ceiling/gamma into a score
was not given verbatim in the task brief, only the parameters. We use the
standard clamped-linear-then-power-law shape, which is the conventional way
to turn (floor, ceiling, gamma) into a bounded [0, 1] score and is the
simplest formula consistent with all stated constraints:

    norm(x; floor, ceiling) = clamp((ceiling - x) / (ceiling - floor), 0, 1)
    score(x; floor, ceiling, gamma) = norm(x; floor, ceiling) ** gamma

This is documented here (and in README.md) so it can be swapped in one place
if the official grader's exact formula turns out to differ.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

TTFT_FLOOR_MS = 10.0
TTFT_CEIL_MS = 400.0
TPOT_FLOOR_MS = 1.0
TPOT_CEIL_MS = 10.0
GAMMA = 2.0
W_TTFT = 0.5
W_TPOT = 0.5


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def metric_score(value_ms: float, floor_ms: float, ceil_ms: float, gamma: float = GAMMA) -> float:
    norm = clamp01((ceil_ms - value_ms) / (ceil_ms - floor_ms))
    return norm ** gamma


def request_score(row: dict) -> float:
    status = row.get("status")
    n_out = row.get("output_tokens") or 0
    if status != "ok" or n_out == 0:
        return 0.0
    ttft_ms = row.get("ttft_ms")
    tpot_ms = row.get("tpot_mean_ms")
    if ttft_ms is None:
        return 0.0
    if tpot_ms is None:
        # Single-token completion: no decode interval to measure. There is
        # no penalty to assess, so treat the (non-existent) TPOT term as
        # best-case rather than dropping the request.
        tpot_ms = TPOT_FLOOR_MS
    s_ttft = metric_score(ttft_ms, TTFT_FLOOR_MS, TTFT_CEIL_MS)
    s_tpot = metric_score(tpot_ms, TPOT_FLOOR_MS, TPOT_CEIL_MS)
    return W_TTFT * s_ttft + W_TPOT * s_tpot


def load_results(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(rows: list[dict], exclude_warmup: bool = True) -> dict:
    scored_rows = [r for r in rows if not (exclude_warmup and r.get("in_warmup"))]
    n = len(scored_rows)
    per_req_scores = [request_score(r) for r in scored_rows]

    errors = [r for r in scored_rows if r.get("status") != "ok"]
    ok_rows = [r for r in scored_rows if r.get("status") == "ok" and (r.get("output_tokens") or 0) > 0]

    ttft_vals = [r["ttft_ms"] for r in ok_rows if r.get("ttft_ms") is not None]
    tpot_vals = [r["tpot_mean_ms"] for r in ok_rows if r.get("tpot_mean_ms") is not None]
    latency_vals = [r["latency_ms"] for r in scored_rows if r.get("latency_ms") is not None]

    summary = {
        "n_requests": n,
        "n_errors": len(errors),
        "error_rate": (len(errors) / n) if n else None,
        "ers_mean": statistics.mean(per_req_scores) if per_req_scores else None,
        "ers_mean_pct": 100.0 * statistics.mean(per_req_scores) if per_req_scores else None,
        "ttft_ms_mean": statistics.mean(ttft_vals) if ttft_vals else None,
        "ttft_ms_p50": percentile(ttft_vals, 0.50) if ttft_vals else None,
        "ttft_ms_p95": percentile(ttft_vals, 0.95) if ttft_vals else None,
        "tpot_ms_mean": statistics.mean(tpot_vals) if tpot_vals else None,
        "tpot_ms_p95": percentile(tpot_vals, 0.95) if tpot_vals else None,
        "latency_ms_mean": statistics.mean(latency_vals) if latency_vals else None,
        "latency_ms_p95": percentile(latency_vals, 0.95) if latency_vals else None,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_jsonl", help="Path to a replay_trace.py .jsonl results file")
    ap.add_argument("--include-warmup", action="store_true", help="Include warmup requests in ERS (default: excluded)")
    ap.add_argument("--json", action="store_true", help="Print summary as JSON instead of text")
    args = ap.parse_args()

    rows = load_results(args.results_jsonl)
    summary = summarize(rows, exclude_warmup=not args.include_warmup)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Requests scored:     {summary['n_requests']}")
    print(f"Errors:              {summary['n_errors']} ({(summary['error_rate'] or 0) * 100:.1f}%)")
    print(f"TTFT mean/p50/p95 (ms): {summary['ttft_ms_mean']:.1f} / {summary['ttft_ms_p50']:.1f} / {summary['ttft_ms_p95']:.1f}" if summary['ttft_ms_mean'] is not None else "TTFT: n/a")
    print(f"TPOT mean / p95 (ms):{summary['tpot_ms_mean']:.2f} / {summary['tpot_ms_p95']:.2f}" if summary['tpot_ms_mean'] is not None else "TPOT: n/a")
    print(f"Latency mean/p95 (ms): {summary['latency_ms_mean']:.1f} / {summary['latency_ms_p95']:.1f}" if summary['latency_ms_mean'] is not None else "Latency: n/a")
    print(f"ERS (mean, 0-100):    {summary['ers_mean_pct']:.2f}" if summary['ers_mean_pct'] is not None else "ERS: n/a")


if __name__ == "__main__":
    main()
