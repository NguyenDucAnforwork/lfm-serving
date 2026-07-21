#!/usr/bin/env python3
"""Compare generated text between two replay_trace.py result JSONL files.

Use this after running replay_trace.py with --keep-output-text. The default
benchmark artifacts intentionally omit text, so this script fails loudly when
text is unavailable rather than pretending token/length deltas prove semantic
accuracy parity.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["request_id"]] = row
    return rows


def bucket_input_tokens(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n < 2500:
        return "<2500"
    if n < 3500:
        return "2500-3499"
    if n < 4500:
        return "3500-4499"
    return ">=4500"


def answer_format(text: str) -> str:
    s = text.strip()
    if not s:
        return "empty"
    if re.fullmatch(r"[A-D]", s, flags=re.I):
        return "single_letter"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return "numeric"
    if s.startswith("{") or s.startswith("["):
        return "json_like"
    if len(s) < 32:
        return "short_text"
    if "\n" in s:
        return "multiline"
    return "paragraph"


def trunc(text: str, n: int) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--candidate-modules", default="unknown")
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)
    common = sorted(set(base) & set(cand))

    missing_text = [
        rid
        for rid in common
        if "output_text" not in base[rid] or "output_text" not in cand[rid]
    ]
    if missing_text:
        raise SystemExit(
            "output_text missing for compared rows. Re-run both experiments "
            "with replay_trace.py --keep-output-text before semantic diffing. "
            f"First missing request_id={missing_text[0]}"
        )

    changed = []
    groups: dict[str, collections.Counter[str]] = {
        "by_turn": collections.Counter(),
        "by_input_bucket": collections.Counter(),
        "by_output_max": collections.Counter(),
        "by_answer_format_pair": collections.Counter(),
        "by_candidate_modules": collections.Counter(),
    }

    for rid in common:
        b = base[rid]
        c = cand[rid]
        btxt = b.get("output_text") or ""
        ctxt = c.get("output_text") or ""
        if btxt == ctxt:
            continue
        item = {
            "request_id": rid,
            "conv_id": b.get("conv_id"),
            "turn_idx": b.get("turn_idx"),
            "target_in_tokens": b.get("target_in_tokens"),
            "out_tokens_max": b.get("out_tokens_max"),
            "baseline_status": b.get("status"),
            "candidate_status": c.get("status"),
            "baseline_output_tokens": b.get("output_tokens"),
            "candidate_output_tokens": c.get("output_tokens"),
            "baseline_text_len": len(btxt),
            "candidate_text_len": len(ctxt),
            "baseline_format": answer_format(btxt),
            "candidate_format": answer_format(ctxt),
            "baseline_preview": trunc(btxt, 240),
            "candidate_preview": trunc(ctxt, 240),
        }
        changed.append(item)
        groups["by_turn"][str(item["turn_idx"])] += 1
        groups["by_input_bucket"][bucket_input_tokens(item["target_in_tokens"])] += 1
        groups["by_output_max"][str(item["out_tokens_max"])] += 1
        groups["by_answer_format_pair"][
            f"{item['baseline_format']}->{item['candidate_format']}"
        ] += 1
        groups["by_candidate_modules"][args.candidate_modules] += 1

    summary = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "candidate_modules": args.candidate_modules,
        "common_requests": len(common),
        "changed_count": len(changed),
        "unchanged_count": len(common) - len(changed),
        "groups": {k: dict(v) for k, v in groups.items()},
        "examples": changed[: args.max_examples],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    if changed:
        print("\nexamples")
        for ex in changed[: args.max_examples]:
            print(
                f"- {ex['request_id']} turn={ex['turn_idx']} "
                f"in={ex['target_in_tokens']} format="
                f"{ex['baseline_format']}->{ex['candidate_format']}"
            )
            print(f"  baseline:  {ex['baseline_preview']}")
            print(f"  candidate: {ex['candidate_preview']}")


if __name__ == "__main__":
    main()
