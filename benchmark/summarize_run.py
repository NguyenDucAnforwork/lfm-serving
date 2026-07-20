#!/usr/bin/env python3
"""Print benchmark metrics needed by the experiment log."""
from __future__ import annotations

import argparse
import json

import compute_ers


def pct(values: list[float], p: float) -> float | None:
    return compute_ers.percentile(values, p) if values else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_jsonl")
    ap.add_argument("--include-warmup", action="store_true")
    args = ap.parse_args()

    rows = compute_ers.load_results(args.results_jsonl)
    scored = [r for r in rows if args.include_warmup or not r.get("in_warmup")]
    ok = [
        r
        for r in scored
        if r.get("status") == "ok" and (r.get("output_tokens") or 0) > 0
    ]
    tpot = [r["tpot_mean_ms"] for r in ok if r.get("tpot_mean_ms") is not None]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    output_tokens = sum(r.get("output_tokens") or 0 for r in ok)

    starts = [r.get("t_send_epoch") for r in ok if r.get("t_send_epoch") is not None]
    wall = 0.0
    if starts:
        first = min(starts)
        last = max(
            r["t_send_epoch"] + (r.get("latency_ms") or 0) / 1000.0
            for r in ok
            if r.get("t_send_epoch") is not None
        )
        wall = max(0.0, last - first)

    summary = compute_ers.summarize(rows, exclude_warmup=not args.include_warmup)
    summary.update(
        {
            "tpot_ms_p50": pct(tpot, 0.50),
            "tpot_ms_p95": pct(tpot, 0.95),
            "ttft_ms_p50": pct(ttft, 0.50),
            "decode_tokens_per_s": output_tokens / wall if wall else None,
            "successful_requests": len(ok),
            "failed_requests": len(scored) - len(ok),
            "wall_time_s": wall,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
