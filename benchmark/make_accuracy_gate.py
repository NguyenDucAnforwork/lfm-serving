#!/usr/bin/env python3
"""Build the W4 pre-submit accuracy gate JSON from lm-eval result files."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


METRICS = {
    "arc_challenge": "acc_norm,none",
    "gsm8k": "exact_match,flexible-extract",
    "gpqa_diamond_zeroshot": "acc_norm,none",
}


THRESHOLDS = {
    "arc_challenge": 0.015,
    "gsm8k": 0.025,
    "gpqa_diamond_zeroshot": 0.04,
}


def latest_result(root: Path, config: str, task: str) -> Path | None:
    pattern = str(root / config / "**" / "results_*.json")
    matches = []
    for raw in glob.glob(pattern, recursive=True):
        path = Path(raw)
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if task in (data.get("results") or {}):
            matches.append(path)
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def read_score(path: Path, task: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    row = data["results"][task]
    metric = METRICS[task]
    stderr = metric.replace(",none", "_stderr,none").replace(
        "exact_match,flexible-extract", "exact_match_stderr,flexible-extract"
    )
    return {
        "path": str(path),
        "task": task,
        "metric": metric,
        "score": row.get(metric),
        "stderr": row.get(stderr),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results/accuracy_w4_gate")
    ap.add_argument("--baseline-config", default="bf16_h200shape")
    ap.add_argument("--fp8-config", default="fp8_h200shape")
    ap.add_argument("--candidate-config", default="w4a16_gptq_marlin")
    ap.add_argument("--require-gpqa", action="store_true")
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    tasks = ["arc_challenge", "gsm8k", "gpqa_diamond_zeroshot"]
    comparisons: dict[str, Any] = {}
    failures: list[str] = []

    for task in tasks:
        base_path = latest_result(root, args.baseline_config, task)
        fp8_path = latest_result(root, args.fp8_config, task)
        cand_path = latest_result(root, args.candidate_config, task)
        if not base_path or not cand_path:
            if task == "gpqa_diamond_zeroshot" and not args.require_gpqa:
                comparisons[task] = {"skipped": True, "reason": "missing GPQA result and require_gpqa=false"}
                continue
            failures.append(f"missing {task} results for baseline or candidate")
            continue

        base = read_score(base_path, task)
        cand = read_score(cand_path, task)
        fp8 = read_score(fp8_path, task) if fp8_path else None
        delta_vs_bf16 = (base["score"] or 0) - (cand["score"] or 0)
        delta_vs_fp8 = ((fp8["score"] or 0) - (cand["score"] or 0)) if fp8 else None
        threshold = THRESHOLDS[task]
        passed = delta_vs_bf16 <= threshold
        if not passed:
            failures.append(f"{task} delta_vs_bf16={delta_vs_bf16:.4f} > {threshold:.4f}")

        comparisons[task] = {
            "baseline": base,
            "fp8_reference": fp8,
            "candidate": cand,
            "delta_vs_bf16": delta_vs_bf16,
            "delta_vs_fp8": delta_vs_fp8,
            "max_allowed_delta_vs_bf16": threshold,
            "pass": passed,
        }

    output_coherence_format = "manual_required"
    failures.append("manual output coherence/format inspection not recorded")

    result = {
        "pass": False,
        "accuracy_drop_risk": True,
        "baseline_config": args.baseline_config,
        "fp8_reference_config": args.fp8_config,
        "candidate_config": args.candidate_config,
        "comparisons": comparisons,
        "output_coherence_format": output_coherence_format,
        "failures": failures,
        "notes": [
            "Set output_coherence_format='pass', remove the manual inspection failure, and set pass=true only after inspecting W4 outputs.",
            "GPQA may be skipped only when time/access is unavailable; ARC full and GSM8K limit 200 are mandatory.",
        ],
    }
    if not failures or failures == ["manual output coherence/format inspection not recorded"]:
        result["accuracy_drop_risk"] = False

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
