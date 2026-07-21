#!/usr/bin/env python3
"""Compare lm-eval per-sample JSONL outputs by doc_id."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows[str(rec["doc_id"])] = rec
    return rows


def correct(rec: dict[str, Any], metric: str) -> bool:
    return bool(rec.get(metric))


def prompt_len(rec: dict[str, Any]) -> int:
    args = rec.get("arguments") or {}
    text = json.dumps(args, ensure_ascii=False)
    return len(text)


def bucket(n: int) -> str:
    for lo, hi in ((0, 999), (1000, 1999), (2000, 3999), (4000, 7999), (8000, 10**9)):
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}"
    return "unknown"


def answer_format(task: str, rec: dict[str, Any]) -> str:
    if task == "gsm8k":
        resp = rec.get("filtered_resps")
        if isinstance(resp, list) and resp:
            val = str(resp[0])
            if val.strip().lstrip("-").replace(".", "", 1).isdigit():
                return "numeric"
            return "non_numeric"
        return "missing"
    if task == "arc_challenge":
        return "multiple_choice_loglikelihood"
    return "unknown"


def summarize(task: str, baseline: Path, candidate: Path, metric: str, modules: str) -> dict[str, Any]:
    base = load(baseline)
    cand = load(candidate)
    common = sorted(set(base) & set(cand), key=lambda x: int(x))
    regressions = []
    fixes = []
    changes = []
    reg_groups: Counter[tuple[str, str, str]] = Counter()
    fix_groups: Counter[tuple[str, str, str]] = Counter()
    for sid in common:
        b = base[sid]
        c = cand[sid]
        b_ok = correct(b, metric)
        c_ok = correct(c, metric)
        b_resp = b.get("filtered_resps")
        c_resp = c.get("filtered_resps")
        if b_resp != c_resp:
            changes.append(sid)
        item = {
            "doc_id": sid,
            "question": (c.get("doc") or {}).get("question"),
            "target": c.get("target"),
            "baseline_response": b_resp,
            "candidate_response": c_resp,
            "prompt_length": prompt_len(c),
            "answer_format": answer_format(task, c),
            "candidate_modules": modules,
        }
        key = (task, bucket(item["prompt_length"]), item["answer_format"])
        if b_ok and not c_ok:
            regressions.append(item)
            reg_groups[key] += 1
        elif (not b_ok) and c_ok:
            fixes.append(item)
            fix_groups[key] += 1
    return {
        "task": task,
        "metric": metric,
        "n_common": len(common),
        "baseline_correct": sum(correct(base[sid], metric) for sid in common),
        "candidate_correct": sum(correct(cand[sid], metric) for sid in common),
        "changed_response_count": len(changes),
        "regression_count": len(regressions),
        "fix_count": len(fixes),
        "regression_groups": [
            {"task": k[0], "prompt_length_bucket": k[1], "answer_format": k[2], "count": v}
            for k, v in reg_groups.most_common()
        ],
        "fix_groups": [
            {"task": k[0], "prompt_length_bucket": k[1], "answer_format": k[2], "count": v}
            for k, v in fix_groups.most_common()
        ],
        "regressions": regressions,
        "fixes": fixes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--candidate-modules", default="unknown")
    ap.add_argument("--json-out", required=True, type=Path)
    args = ap.parse_args()
    out = summarize(args.task, args.baseline, args.candidate, args.metric, args.candidate_modules)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"{args.task}: common={out['n_common']} baseline_correct={out['baseline_correct']} "
        f"candidate_correct={out['candidate_correct']} regressions={out['regression_count']} "
        f"fixes={out['fix_count']} changed={out['changed_response_count']}"
    )
    for group in out["regression_groups"][:10]:
        print(
            f"  reg {group['count']:>3} prompt={group['prompt_length_bucket']} "
            f"format={group['answer_format']}"
        )


if __name__ == "__main__":
    main()
