#!/usr/bin/env python3
"""Create a pass/fail report for a decode-cost candidate.

The report intentionally separates measured evidence from gates. It is meant to
be run after the H200 experiments complete, using result directories produced by
scripts/run_experiment.sh / scripts/run_profile_capture.sh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(errors="replace")


def run_dir_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path / "summary_ext.json") or {}
    validation = load_json(path / "w4_validation.json")
    profile = load_json(path / "profile_buckets.json")
    failure_analysis = load_text(path / "failure_analysis.txt")
    vram_peak = None
    vram_path = path / "vram_peak_mib.txt"
    if vram_path.exists():
        raw = vram_path.read_text().strip()
        if raw.isdigit():
            vram_peak = int(raw)
    config = load_text(path / "config.env")
    return {
        "path": str(path),
        "has_failed_marker": (path / "FAILED").exists(),
        "has_contention_marker": (path / "CONTENDED").exists(),
        "summary": summary,
        "w4_validation": validation,
        "profile": profile,
        "failure_analysis_present": failure_analysis is not None,
        "vram_peak_mib": vram_peak,
        "config_env": config,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compose_flags_valid(path: Path | None, candidate: str) -> tuple[bool, list[str]]:
    if path is None or not path.exists():
        return False, ["compose file missing"]
    text = path.read_text()
    required = [
        "vllm.entrypoints.openai.api_server",
        "--model=/model",
        "--served-model-name=LFM2.5-1.2B-Instruct",
        "--dtype=bfloat16",
        "--max-model-len=5120",
        "--gpu-memory-utilization=0.70",
        "--max-num-seqs=8",
        "--max-num-batched-tokens=512",
        "--enable-prefix-caching",
        "--prefix-caching-hash-algo=xxhash",
    ]
    failures = [f"missing {token}" for token in required if token not in text]
    lower = text.lower()
    if "bitsandbytes" in lower:
        failures.append("mentions BitsAndBytes")
    if "w4" in candidate or "gptq" in candidate:
        if "--quantization=compressed-tensors" not in text:
            failures.append("missing --quantization=compressed-tensors")
        if "--linear-backend=machete" not in text and "--linear-backend=marlin" not in text:
            failures.append("missing --linear-backend=machete or --linear-backend=marlin")
        if "fp8_per_tensor" in text or "--kv-cache-dtype=fp8" in text or "--kv-cache-dtype=fp8_e5m2" in text:
            failures.append("mixes W4 with FP8")
    return not failures, failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="Result directories")
    ap.add_argument("--baseline-name", default="official_fp8_per_tensor_s2")
    ap.add_argument("--baseline-tbt-p50-ms", type=float, default=4.0)
    ap.add_argument("--baseline-failed-count", type=int, default=4)
    ap.add_argument("--accuracy-diff-json", default=None)
    ap.add_argument("--output-diff-json", default=None)
    ap.add_argument("--docker-compose", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    runs = [run_dir_summary(Path(p)) for p in args.runs]
    tbt_p50 = [
        r["summary"].get("tpot_ms_p50")
        for r in runs
        if isinstance(r["summary"].get("tpot_ms_p50"), (int, float))
    ]
    failed = [
        r["summary"].get("failed_requests")
        for r in runs
        if isinstance(r["summary"].get("failed_requests"), (int, float))
    ]
    ers = [
        r["summary"].get("ers_mean_pct")
        for r in runs
        if isinstance(r["summary"].get("ers_mean_pct"), (int, float))
    ]
    ttft_p50 = [
        r["summary"].get("ttft_ms_p50")
        for r in runs
        if isinstance(r["summary"].get("ttft_ms_p50"), (int, float))
    ]
    vram = [r["vram_peak_mib"] for r in runs if isinstance(r["vram_peak_mib"], int)]

    accuracy = load_json(Path(args.accuracy_diff_json)) if args.accuracy_diff_json else None
    output_diff = load_json(Path(args.output_diff_json)) if args.output_diff_json else None
    compose_path = Path(args.docker_compose) if args.docker_compose else None
    compose_ok, compose_failures = compose_flags_valid(compose_path, args.candidate)

    worst_improvement = None
    if tbt_p50 and args.baseline_tbt_p50_ms:
        worst_improvement = min((args.baseline_tbt_p50_ms - v) / args.baseline_tbt_p50_ms for v in tbt_p50)

    w4_validations = [r["w4_validation"] for r in runs if r["w4_validation"] is not None]
    clean_runs = [
        r
        for r in runs
        if not r["has_failed_marker"]
        and not r["has_contention_marker"]
        and isinstance(r["summary"].get("tpot_ms_p50"), (int, float))
    ]

    gates = {
        "at_least_3_clean_runs": len(clean_runs) >= 3,
        "tbt_target_met": bool(tbt_p50) and max(tbt_p50) <= 3.5,
        "decode_improvement_met": worst_improvement is not None and worst_improvement >= 0.15,
        "no_extra_failures": bool(failed) and max(failed) <= args.baseline_failed_count,
        "w4_backend_and_artifact_valid": not w4_validations or all(v.get("valid") is True for v in w4_validations),
        "accuracy_not_worse": accuracy is not None
        and accuracy.get("candidate_correct", -1) >= accuracy.get("baseline_correct", 10**9),
        "submission_compose_present": compose_path is not None and compose_path.exists(),
        "submission_compose_flags_valid": compose_ok,
    }
    gates["decode_gate_met"] = gates["tbt_target_met"] or gates["decode_improvement_met"]
    gates["candidate_passes"] = (
        gates["at_least_3_clean_runs"]
        and gates["decode_gate_met"]
        and gates["no_extra_failures"]
        and gates["w4_backend_and_artifact_valid"]
        and gates["accuracy_not_worse"]
        and gates["submission_compose_present"]
        and gates["submission_compose_flags_valid"]
    )

    report = {
        "candidate": args.candidate,
        "baseline": {
            "name": args.baseline_name,
            "tbt_p50_ms": args.baseline_tbt_p50_ms,
            "failed_count": args.baseline_failed_count,
        },
        "runs": runs,
        "accuracy_diff": accuracy,
        "output_diff": output_diff,
        "docker_compose": str(compose_path) if compose_path else None,
        "docker_compose_failures": compose_failures,
        "metrics": {
            "clean_run_count": len(clean_runs),
            "tbt_p50_ms_min": min(tbt_p50) if tbt_p50 else None,
            "tbt_p50_ms_max": max(tbt_p50) if tbt_p50 else None,
            "tbt_p50_worst_improvement_frac": worst_improvement,
            "ers_min": min(ers) if ers else None,
            "ers_max": max(ers) if ers else None,
            "ttft_p50_min": min(ttft_p50) if ttft_p50 else None,
            "ttft_p50_max": max(ttft_p50) if ttft_p50 else None,
            "failed_requests_max": max(failed) if failed else None,
            "vram_peak_mib_max": max(vram) if vram else None,
        },
        "gates": gates,
    }

    lines = [
        f"# Candidate report: {args.candidate}",
        "",
        f"Baseline: {args.baseline_name}, TBT p50 {args.baseline_tbt_p50_ms} ms, "
        f"failed_count {args.baseline_failed_count}.",
        "",
        "## Decision",
        "",
        "PASS" if gates["candidate_passes"] else "FAIL / INCOMPLETE",
        "",
        "## Metrics",
        "",
        f"- runs: {len(runs)}",
        f"- clean runs: {len(clean_runs)}",
        f"- TBT p50 min/max: {fmt(report['metrics']['tbt_p50_ms_min'])} / {fmt(report['metrics']['tbt_p50_ms_max'])} ms",
        f"- worst TBT p50 improvement: {fmt((worst_improvement or 0.0) * 100, 2)}%",
        f"- ERS min/max: {fmt(report['metrics']['ers_min'], 2)} / {fmt(report['metrics']['ers_max'], 2)}",
        f"- TTFT p50 min/max: {fmt(report['metrics']['ttft_p50_min'])} / {fmt(report['metrics']['ttft_p50_max'])} ms",
        f"- max failed requests: {fmt(report['metrics']['failed_requests_max'])}",
        f"- max VRAM peak: {fmt(report['metrics']['vram_peak_mib_max'])} MiB",
        "",
        "## Gates",
        "",
    ]
    for key, value in gates.items():
        lines.append(f"- {key}: {value}")
    if compose_failures:
        lines.append("")
        lines.append("Compose validation failures:")
        for failure in compose_failures:
            lines.append(f"- {failure}")

    if accuracy is not None:
        lines.extend(
            [
                "",
                "## Accuracy",
                "",
                f"- baseline_correct: {accuracy.get('baseline_correct')}",
                f"- candidate_correct: {accuracy.get('candidate_correct')}",
                f"- regressions: {accuracy.get('regression_count')}",
                f"- fixes: {accuracy.get('fix_count')}",
                f"- changed outputs: {accuracy.get('changed_output_count')}",
            ]
        )

    if output_diff is not None:
        lines.extend(
            [
                "",
                "## Trace output diff",
                "",
                f"- common requests: {output_diff.get('common_requests')}",
                f"- changed outputs: {output_diff.get('changed_count')}",
                f"- unchanged outputs: {output_diff.get('unchanged_count')}",
            ]
        )
        groups = output_diff.get("groups") or {}
        for group_name in ("by_turn", "by_input_bucket", "by_answer_format_pair"):
            group = groups.get(group_name)
            if not group:
                continue
            lines.append(f"- {group_name}: {group}")

    md = "\n".join(lines) + "\n"
    print(md)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)


if __name__ == "__main__":
    main()
