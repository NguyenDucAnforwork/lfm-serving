#!/usr/bin/env python3
"""Build a consolidated decode-cost handoff report.

This is for the case where no local candidate passes. It combines:

- matched FP8 baseline summary;
- corrected FP8 profile bucket attribution;
- W4 candidate report / aggregate evidence;
- the concrete next step allowed by the profile evidence.

The output is intentionally conservative: it does not mark success unless the
candidate report gates already prove success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bucket_pct(profile: dict[str, Any], name: str) -> float:
    return float(((profile.get("buckets") or {}).get(name) or {}).get("pct") or 0.0)


def bucket_ms(profile: dict[str, Any], name: str) -> float:
    return float(((profile.get("buckets") or {}).get(name) or {}).get("duration_ms") or 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-summary", required=True, type=Path)
    ap.add_argument("--profile-buckets", required=True, type=Path)
    ap.add_argument("--kernel-next-step", required=True, type=Path)
    ap.add_argument("--candidate-report", required=True, type=Path)
    ap.add_argument("--aggregate", type=Path)
    ap.add_argument("--md-out", required=True, type=Path)
    ap.add_argument("--json-out", required=True, type=Path)
    args = ap.parse_args()

    baseline = load_json(args.baseline_summary)
    profile = load_json(args.profile_buckets)
    next_step = load_json(args.kernel_next_step)
    candidate = load_json(args.candidate_report)
    aggregate = load_json(args.aggregate) if args.aggregate and args.aggregate.exists() else {}

    gates = candidate.get("gates") or {}
    cand_metrics = candidate.get("metrics") or {}
    official_baseline = candidate.get("baseline") or {}

    done_criteria = {
        "decode_target_or_15pct_improvement": bool(
            gates.get("tbt_target_met") or gates.get("decode_improvement_met")
        ),
        "no_additional_failures": bool(gates.get("no_extra_failures")),
        "accuracy_parity": bool(gates.get("accuracy_not_worse")),
        "three_clean_runs": bool(gates.get("at_least_3_clean_runs")),
        "candidate_passes": bool(gates.get("candidate_passes")),
    }

    rejected_hypotheses: list[str] = []
    if bucket_pct(profile, "fp8_scaling") < 10.0:
        rejected_hypotheses.append(
            "Static/offline FP8 is not justified by this profile: fp8_scaling "
            f"is {bucket_pct(profile, 'fp8_scaling'):.2f}%."
        )
    if bucket_pct(profile, "gated_conv_state") < 15.0:
        rejected_hypotheses.append(
            "Gated-conv/state fusion is not justified by this profile: "
            f"gated_conv_state is {bucket_pct(profile, 'gated_conv_state'):.2f}%."
        )
    if not gates.get("decode_gate_met"):
        rejected_hypotheses.append(
            "Local W4A16 Marlin no-conv is rejected: it did not meet the TBT "
            "target or 15% improvement gate."
        )

    profile_backed_kernel_next_step = (
        "Implement or enable an H200 decode grouped-GEMM / launch-fusion path for "
        "LFM2's hot Linear shapes (hidden=2048, MLP intermediate=8192 after "
        "auto-adjust), covering attention and MLP W4A16/FP8 linears while leaving "
        "ShortConv projections BF16. The target is to reduce the combined GEMM "
        f"({bucket_pct(profile, 'gemm'):.2f}%) and CUDA launch/runtime "
        f"({bucket_pct(profile, 'launch_overhead'):.2f}%) buckets without touching "
        "the non-hot FP8 scaling or gated-conv paths."
    )

    report = {
        "official_gate_baseline": {
            "name": official_baseline.get("name"),
            "tbt_p50_ms": official_baseline.get("tbt_p50_ms"),
            "failed_count": official_baseline.get("failed_count"),
        },
        "baseline_summary": {
            "ers_mean_pct": baseline.get("ers_mean_pct"),
            "tbt_ms_mean": baseline.get("tpot_ms_mean"),
            "tbt_ms_p50": baseline.get("tpot_ms_p50"),
            "tbt_ms_p95": baseline.get("tpot_ms_p95"),
            "ttft_ms_mean": baseline.get("ttft_ms_mean"),
            "failed_requests": baseline.get("failed_requests"),
        },
        "profile_buckets": {
            name: {"duration_ms": bucket_ms(profile, name), "pct": bucket_pct(profile, name)}
            for name in (
                "gemm",
                "launch_overhead",
                "attention",
                "gated_conv_state",
                "fp8_scaling",
                "other",
            )
        },
        "candidate_metrics": cand_metrics,
        "candidate_gates": gates,
        "aggregate_gate_checks": aggregate.get("gate_checks") or {},
        "done_criteria": done_criteria,
        "rejected_hypotheses": rejected_hypotheses,
        "recommended_area": next_step.get("recommended_area"),
        "pre_w4_recommended_next_step": next_step.get("recommended_next_step"),
        "profile_backed_kernel_next_step": profile_backed_kernel_next_step,
    }

    lines = [
        "# Decode-cost handoff report",
        "",
        "## Decision",
        "",
        "PASS" if done_criteria["candidate_passes"] else "NO PASSING CANDIDATE",
        "",
        "## Official gate baseline",
        "",
        f"- name: {fmt(official_baseline.get('name'))}",
        f"- TBT p50: {fmt(official_baseline.get('tbt_p50_ms'))} ms",
        f"- failed_count: {fmt(official_baseline.get('failed_count'))}",
        "",
        "## Matched local FP8 baseline",
        "",
        f"- ERS: {fmt(baseline.get('ers_mean_pct'), 2)}",
        f"- TBT mean/p50/p95: {fmt(baseline.get('tpot_ms_mean'))} / "
        f"{fmt(baseline.get('tpot_ms_p50'))} / {fmt(baseline.get('tpot_ms_p95'))} ms",
        f"- TTFT mean: {fmt(baseline.get('ttft_ms_mean'))} ms",
        f"- failures: {fmt(baseline.get('failed_requests'))}",
        "",
        "## Corrected FP8 profile buckets",
        "",
        "| bucket | duration ms | share |",
        "|---|---:|---:|",
    ]
    for name, values in report["profile_buckets"].items():
        lines.append(f"| {name} | {values['duration_ms']:.3f} | {values['pct']:.2f}% |")

    lines.extend(
        [
            "",
            "## W4 candidate status",
            "",
            f"- TBT p50 min/max: {fmt(cand_metrics.get('tbt_p50_ms_min'))} / "
            f"{fmt(cand_metrics.get('tbt_p50_ms_max'))} ms",
            f"- worst p50 improvement: "
            f"{fmt((cand_metrics.get('tbt_p50_worst_improvement_frac') or 0.0) * 100, 2)}%",
            f"- clean runs: {fmt(cand_metrics.get('clean_run_count'))}",
            f"- max failures: {fmt(cand_metrics.get('failed_requests_max'))}",
            "",
            "## Done criteria audit",
            "",
        ]
    )
    for key, value in done_criteria.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Rejected hypotheses", ""])
    for item in rejected_hypotheses:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Recommended next step",
            "",
            f"- area: {next_step.get('recommended_area')} / grouped_decode_gemm_launch_fusion",
            f"- immediate H200 validation: {next_step.get('recommended_next_step')}",
            f"- concrete kernel step if no candidate passes: {profile_backed_kernel_next_step}",
            "",
            "This is not a completion claim. H200/Machete evidence is still required "
            "before any W4 submission candidate can pass.",
        ]
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.md_out}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
