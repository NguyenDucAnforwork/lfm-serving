#!/usr/bin/env python3
"""Compare per-sample accuracy JSONL files and group exact regressions/fixes.

Designed for outputs from benchmark/gpqa_eval.py, but tolerant of older records
that only contain domain/expected/got/correct/status.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_id(record: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "question_id", "id"):
        if record.get(key) is not None:
            return str(record[key])
    return str(index)


def length_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    v = int(value)
    for lo, hi in ((0, 999), (1000, 1999), (2000, 3999), (4000, 7999), (8000, 10**9)):
        if lo <= v <= hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    return "unknown"


def answer_format(record: dict[str, Any]) -> str:
    if record.get("answer_format"):
        return str(record["answer_format"])
    if record.get("got") is None:
        return "unparsed"
    return "parsed_letter"


def build_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {sample_id(rec, i): rec for i, rec in enumerate(records)}


def group_key(record: dict[str, Any], module_policy: str) -> tuple[str, str, str, str]:
    return (
        str(record.get("task") or record.get("domain") or "unknown"),
        length_bucket(record.get("prompt_tokens_est") or record.get("problem_len")),
        answer_format(record),
        module_policy,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="Baseline per-sample JSONL")
    ap.add_argument("--candidate", required=True, help="Candidate per-sample JSONL")
    ap.add_argument(
        "--candidate-modules",
        default="unknown",
        help="Quantized/kept-BF16 module policy label to include in grouping",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = build_index(load_jsonl(args.baseline))
    cand = build_index(load_jsonl(args.candidate))
    ids = sorted(set(base) & set(cand), key=lambda x: (len(x), x))

    regressions = []
    fixes = []
    changed_outputs = []
    missing = sorted(set(base) ^ set(cand), key=lambda x: (len(x), x))

    for sid in ids:
        b = base[sid]
        c = cand[sid]
        if b.get("got") != c.get("got") or b.get("status") != c.get("status"):
            changed_outputs.append({"sample_id": sid, "baseline": b, "candidate": c})
        if b.get("correct") is True and c.get("correct") is not True:
            regressions.append({"sample_id": sid, "baseline": b, "candidate": c})
        elif b.get("correct") is not True and c.get("correct") is True:
            fixes.append({"sample_id": sid, "baseline": b, "candidate": c})

    reg_groups: Counter[tuple[str, str, str, str]] = Counter()
    fix_groups: Counter[tuple[str, str, str, str]] = Counter()
    status_pairs = Counter()
    for item in regressions:
        reg_groups[group_key(item["candidate"], args.candidate_modules)] += 1
        status_pairs[(str(item["baseline"].get("status")), str(item["candidate"].get("status")))] += 1
    for item in fixes:
        fix_groups[group_key(item["candidate"], args.candidate_modules)] += 1

    summary = {
        "n_common": len(ids),
        "n_missing_or_extra": len(missing),
        "baseline_correct": sum(1 for sid in ids if base[sid].get("correct") is True),
        "candidate_correct": sum(1 for sid in ids if cand[sid].get("correct") is True),
        "changed_output_count": len(changed_outputs),
        "regression_count": len(regressions),
        "fix_count": len(fixes),
        "missing_or_extra_sample_ids": missing,
        "regression_groups": [
            {
                "task_or_domain": k[0],
                "prompt_length_bucket": k[1],
                "answer_format": k[2],
                "candidate_modules": k[3],
                "count": v,
            }
            for k, v in reg_groups.most_common()
        ],
        "fix_groups": [
            {
                "task_or_domain": k[0],
                "prompt_length_bucket": k[1],
                "answer_format": k[2],
                "candidate_modules": k[3],
                "count": v,
            }
            for k, v in fix_groups.most_common()
        ],
        "regression_status_pairs": [
            {"baseline_status": k[0], "candidate_status": k[1], "count": v}
            for k, v in status_pairs.most_common()
        ],
        "regressions": regressions,
        "fixes": fixes,
        "changed_outputs": changed_outputs,
    }

    print(
        "Accuracy compare: "
        f"common={summary['n_common']} "
        f"baseline_correct={summary['baseline_correct']} "
        f"candidate_correct={summary['candidate_correct']} "
        f"regressions={summary['regression_count']} "
        f"fixes={summary['fix_count']} "
        f"changed_outputs={summary['changed_output_count']}"
    )
    if summary["regression_groups"]:
        print("Regression groups:")
        for group in summary["regression_groups"]:
            print(
                "  "
                f"{group['count']:>3}  task={group['task_or_domain']} "
                f"prompt={group['prompt_length_bucket']} "
                f"format={group['answer_format']} "
                f"modules={group['candidate_modules']}"
            )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
