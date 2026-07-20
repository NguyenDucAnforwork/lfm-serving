#!/usr/bin/env python3
"""Cluster replay_trace failures by type, turn, length, concurrency, and time."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify(row: dict, server_had_cuda_error: bool = False) -> str:
    status = row.get("status")
    err = (row.get("error") or "").lower()
    if status == "timeout" or "timeout" in err:
        return "timeout"
    if "out of memory" in err or "oom" in err:
        return "oom"
    if "cuda" in err or "illegal memory" in err:
        return "cuda"
    if server_had_cuda_error and (
        "enginecore encountered" in err or "clientconnectorerror" in err
    ):
        return "cuda_after_engine_crash"
    if "trunc" in err or row.get("status") == "empty_output":
        return "malformed_or_truncated_output"
    if status and status != "ok":
        return status
    return "ok"


def active_concurrency(rows: list[dict], row: dict) -> int:
    start = row.get("t_send_epoch")
    latency = row.get("latency_ms")
    if start is None or latency is None:
        return -1
    end = start + latency / 1000.0
    n = 0
    for other in rows:
        os = other.get("t_send_epoch")
        ol = other.get("latency_ms")
        if os is None or ol is None:
            continue
        oe = os + ol / 1000.0
        if os <= end and oe >= start:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_jsonl")
    ap.add_argument("--server-log")
    args = ap.parse_args()

    rows = load_rows(Path(args.results_jsonl))
    server_had_cuda_error = False
    if args.server_log:
        log_text = Path(args.server_log).read_text(errors="replace").lower()
        server_had_cuda_error = "cuda error" in log_text or "illegal memory" in log_text
    failures = [r for r in rows if r.get("status") != "ok"]
    print(f"total_rows={len(rows)} failures={len(failures)}")
    print(
        "by_type",
        dict(Counter(classify(r, server_had_cuda_error) for r in failures)),
    )
    print("by_turn", dict(Counter(r.get("turn_idx") for r in failures)))

    by_len = Counter()
    for r in failures:
        in_bucket = 500 * round((r.get("target_in_tokens") or 0) / 500)
        out_bucket = 100 * round((r.get("out_tokens_max") or 0) / 100)
        by_len[(in_bucket, out_bucket)] += 1
    print("by_input_output_len_bucket", dict(by_len))

    print(
        "by_observed_concurrency",
        dict(Counter(active_concurrency(rows, r) for r in failures)),
    )

    by_minute = Counter()
    starts = [r.get("t_send_epoch") for r in rows if r.get("t_send_epoch") is not None]
    first = min(starts) if starts else None
    for r in failures:
        t = r.get("t_send_epoch")
        if t is not None and first is not None:
            by_minute[int((t - first) // 60)] += 1
    print("by_elapsed_minute", dict(by_minute))

    examples = defaultdict(list)
    for r in failures:
        examples[classify(r, server_had_cuda_error)].append(
            {
                "request_id": r.get("request_id"),
                "turn_idx": r.get("turn_idx"),
                "target_in_tokens": r.get("target_in_tokens"),
                "out_tokens_max": r.get("out_tokens_max"),
                "error": r.get("error"),
            }
        )
    print("examples")
    for kind, vals in examples.items():
        print(kind, vals[:5])


if __name__ == "__main__":
    main()
