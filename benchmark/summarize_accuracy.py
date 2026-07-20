#!/usr/bin/env python3
"""Summarize per-sample accuracy JSONL from benchmark/gpqa_eval.py."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = load_rows(Path(args.jsonl))
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct") is True)
    errors = sum(1 for r in rows if r.get("status") != "ok")
    unparsed = sum(1 for r in rows if r.get("answer_format") in ("empty", "unparsed"))
    summary = {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "errors": errors,
        "unparsed": unparsed,
        "by_domain": dict(Counter(str(r.get("domain") or "unknown") for r in rows)),
        "correct_by_domain": dict(Counter(str(r.get("domain") or "unknown") for r in rows if r.get("correct") is True)),
        "by_status": dict(Counter(str(r.get("status") or "unknown") for r in rows)),
        "by_answer_format": dict(Counter(str(r.get("answer_format") or "unknown") for r in rows)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
