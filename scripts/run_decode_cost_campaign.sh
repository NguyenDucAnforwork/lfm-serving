#!/bin/bash
# Run the constrained decode-cost campaign on an H200/CUDA-13-capable host.
#
# This encodes the goal order:
#   1. profile matched fp8_per_tensor baseline;
#   2. test W4A16 GPTQ compressed-tensors with Machete first, Marlin fallback;
#   3. produce aggregation/report artifacts.
#
# It deliberately does not run BitsAndBytes, does not combine W4 with FP8, and
# does not launch scheduler/cache sweeps. Static FP8 or custom fusion work must
# be triggered from profile evidence, not from this campaign script.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TRACE="${TRACE:-trace_grading_spec.jsonl}"
WORKLOAD="${WORKLOAD:-spec}"
MODEL_NAME="${MODEL_NAME:-LFM2.5-1.2B-Instruct}"
W4_ARTIFACT="${W4_ARTIFACT:-artifacts/lfm2-w4a16-gptq-g128-no-conv}"
W4_RUNS="${W4_RUNS:-3}"
DO_QUANTIZE="${DO_QUANTIZE:-1}"
DO_PROFILE="${DO_PROFILE:-1}"
DO_ACCURACY="${DO_ACCURACY:-0}"
DO_OUTPUT_DIFF="${DO_OUTPUT_DIFF:-0}"
BASELINE_TBT_P50_MS="${BASELINE_TBT_P50_MS:-4.0}"
BASELINE_FAILED_COUNT="${BASELINE_FAILED_COUNT:-4}"
REQUIRE_H200="${REQUIRE_H200:-1}"

stamp="$(date +%Y%m%d-%H%M%S)"
campaign_dir="$PROJECT_DIR/results/decode_cost_campaign_$stamp"
mkdir -p "$campaign_dir"

echo "Campaign dir: $campaign_dir"
echo "TRACE=$TRACE WORKLOAD=$WORKLOAD MODEL_NAME=$MODEL_NAME"
echo "W4_ARTIFACT=$W4_ARTIFACT W4_RUNS=$W4_RUNS"
echo "DO_PROFILE=$DO_PROFILE DO_QUANTIZE=$DO_QUANTIZE DO_ACCURACY=$DO_ACCURACY DO_OUTPUT_DIFF=$DO_OUTPUT_DIFF REQUIRE_H200=$REQUIRE_H200"

if [[ "$REQUIRE_H200" == "1" ]]; then
  pybin="$PROJECT_DIR/.venv/bin/python"
  [[ -x "$pybin" ]] || pybin="python3"
  if ! "$pybin" scripts/check_gpu_capability.py \
    --min-major 9 \
    --json-out "$campaign_dir/gpu_capability_check.json"; then
    echo "This campaign is intended for H200/sm90 Machete validation." >&2
    echo "Current GPU does not satisfy compute capability >= 9.0." >&2
    echo "Set REQUIRE_H200=0 only for local smoke/fallback runs that will not be treated as H200 evidence." >&2
    exit 1
  fi
fi

if [[ "$W4_ARTIFACT" != "artifacts/lfm2-w4a16-gptq-g128-no-conv" ]]; then
  echo "This campaign uses configs/w4a16_gptq_{machete,marlin}.env, which point at artifacts/lfm2-w4a16-gptq-g128-no-conv." >&2
  echo "Either use that artifact path or update the configs before running." >&2
  exit 1
fi

if [[ "$DO_PROFILE" == "1" ]]; then
  profile_name="fp8_profile_spec_$stamp"
  TRACE="$TRACE" WORKLOAD="$WORKLOAD" MODEL_NAME="$MODEL_NAME" \
    bash scripts/run_profile_capture.sh fp8_h200shape "$profile_name"
  ln -s "../$profile_name" "$campaign_dir/fp8_profile" 2>/dev/null || true
  if [[ -f "results/$profile_name/profile_buckets.json" ]]; then
    python3 benchmark/recommend_kernel_step.py \
      "results/$profile_name/profile_buckets.json" \
      --json-out "$campaign_dir/kernel_next_step.json" \
      > "$campaign_dir/kernel_next_step.txt"
  fi
fi

if [[ "$DO_QUANTIZE" == "1" && ! -d "$W4_ARTIFACT" ]]; then
  if [[ -f "$PROJECT_DIR/.venv-quant/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv-quant/bin/activate"
  fi
  python3 scripts/quantize_w4a16_gptq.py \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --recipe recipes/w4a16_gptq_group128_no_conv.yaml \
    --output-dir "$W4_ARTIFACT" \
    --num-calibration-samples 256 \
    --max-seq-length 2048
fi

if [[ ! -d "$W4_ARTIFACT" ]]; then
  echo "Missing W4 artifact at $W4_ARTIFACT. Set DO_QUANTIZE=1 or provide W4_ARTIFACT." >&2
  exit 1
fi

python3 scripts/validate_w4_candidate.py --artifact "$W4_ARTIFACT" \
  > "$campaign_dir/w4_artifact_validation.json"

run_backend() {
  local config_name="$1"
  local backend_file="$2"
  local had_failure=0
  : > "$backend_file"
  for i in $(seq 1 "$W4_RUNS"); do
    echo "Running $config_name repeat $i/$W4_RUNS"
    if TRACE="$TRACE" WORKLOAD="$WORKLOAD" MODEL_NAME="$MODEL_NAME" \
      bash scripts/run_experiment.sh "$config_name"; then
      latest="$(ls -td results/${config_name}_* 2>/dev/null | head -1 || true)"
      if [[ -n "$latest" ]]; then
        echo "$latest" >> "$backend_file"
      fi
    else
      had_failure=1
      latest="$(ls -td results/${config_name}_* 2>/dev/null | head -1 || true)"
      if [[ -n "$latest" ]]; then
        echo "$latest" >> "$backend_file"
      fi
      break
    fi
  done
  return "$had_failure"
}

machete_runs="$campaign_dir/w4_machete_runs.txt"
marlin_runs="$campaign_dir/w4_marlin_runs.txt"
if ! run_backend w4a16_gptq_machete "$machete_runs"; then
  echo "Machete run failed; running Marlin fallback."
  run_backend w4a16_gptq_marlin "$marlin_runs" || true
fi

if [[ -s "$machete_runs" ]]; then
  python3 benchmark/aggregate_runs.py $(tr '\n' ' ' < "$machete_runs") \
    --baseline-tbt-p50-ms "$BASELINE_TBT_P50_MS" \
    --baseline-failed-requests "$BASELINE_FAILED_COUNT" \
    --json-out "$campaign_dir/w4_machete_aggregate.json" \
    > "$campaign_dir/w4_machete_aggregate.txt" || true
fi

if [[ ! -s "$marlin_runs" ]] && ! python3 - <<'PY' "$campaign_dir/w4_machete_aggregate.json"
import json, sys
try:
    g = json.load(open(sys.argv[1]))["gate_checks"]
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if (g["all_tbt_p50_le_3p5_ms"] or g["all_runs_ge_15pct_tbt_p50_improvement"]) and g["no_extra_failures_vs_baseline"] else 1)
PY
then
  echo "Machete did not meet decode/failure gates; running Marlin fallback."
  run_backend w4a16_gptq_marlin "$marlin_runs" || true
fi

if [[ -s "$marlin_runs" ]]; then
  python3 benchmark/aggregate_runs.py $(tr '\n' ' ' < "$marlin_runs") \
    --baseline-tbt-p50-ms "$BASELINE_TBT_P50_MS" \
    --baseline-failed-requests "$BASELINE_FAILED_COUNT" \
    --json-out "$campaign_dir/w4_marlin_aggregate.json" \
    > "$campaign_dir/w4_marlin_aggregate.txt" || true
fi

accuracy_diff=""
output_diff=""

chosen_runs_file="$machete_runs"
candidate="w4a16_gptq_machete"
candidate_config="w4a16_gptq_machete"
if [[ -s "$marlin_runs" ]]; then
  chosen_runs_file="$marlin_runs"
  candidate="w4a16_gptq_marlin"
  candidate_config="w4a16_gptq_marlin"
fi

if [[ "$DO_ACCURACY" == "1" ]]; then
  echo "Running GPQA accuracy for FP8 baseline and $candidate_config"
  bash scripts/run_accuracy.sh fp8_h200shape "campaign_fp8_$stamp"
  fp8_acc_run="$(ls -td results/fp8_h200shape_campaign_fp8_${stamp}_* 2>/dev/null | head -1)"
  bash scripts/run_accuracy.sh "$candidate_config" "campaign_${candidate_config}_$stamp"
  w4_acc_run="$(ls -td results/${candidate_config}_campaign_${candidate_config}_${stamp}_* 2>/dev/null | head -1)"
  echo "$fp8_acc_run" > "$campaign_dir/fp8_accuracy_run.txt"
  echo "$w4_acc_run" > "$campaign_dir/w4_accuracy_run.txt"
  python3 benchmark/compare_accuracy.py \
    --baseline "$fp8_acc_run/accuracy/gpqa.jsonl" \
    --candidate "$w4_acc_run/accuracy/gpqa.jsonl" \
    --candidate-modules attention_mlp_w4_g128_short_conv_lm_head_bf16 \
    --json-out "$campaign_dir/accuracy_diff.json" \
    > "$campaign_dir/accuracy_diff.txt"
fi

if [[ -f "$campaign_dir/accuracy_diff.json" ]]; then
  accuracy_diff="$campaign_dir/accuracy_diff.json"
fi

if [[ "$DO_OUTPUT_DIFF" == "1" ]]; then
  echo "Running paired trace output capture for FP8 baseline and $candidate_config"
  TRACE="$TRACE" WORKLOAD="$WORKLOAD" MODEL_NAME="$MODEL_NAME" \
    bash scripts/run_experiment.sh fp8_h200shape --keep-output-text
  fp8_text_run="$(ls -td results/fp8_h200shape_* 2>/dev/null | head -1)"
  TRACE="$TRACE" WORKLOAD="$WORKLOAD" MODEL_NAME="$MODEL_NAME" \
    bash scripts/run_experiment.sh "$candidate_config" --keep-output-text
  w4_text_run="$(ls -td results/${candidate_config}_* 2>/dev/null | head -1)"
  echo "$fp8_text_run" > "$campaign_dir/fp8_output_text_run.txt"
  echo "$w4_text_run" > "$campaign_dir/w4_output_text_run.txt"
  python3 benchmark/compare_run_outputs.py \
    --baseline "$fp8_text_run/results.jsonl" \
    --candidate "$w4_text_run/results.jsonl" \
    --candidate-modules attention_mlp_w4_g128_short_conv_lm_head_bf16 \
    --json-out "$campaign_dir/output_diff.json" \
    > "$campaign_dir/output_diff.txt"
fi

if [[ -f "$campaign_dir/output_diff.json" ]]; then
  output_diff="$campaign_dir/output_diff.json"
fi

report_args=(
  --candidate "$candidate"
  --runs
)
# shellcheck disable=SC2207
chosen_runs=($(tr '\n' ' ' < "$chosen_runs_file"))
if [[ "${#chosen_runs[@]}" -eq 0 ]]; then
  echo "No W4 run directories were produced; campaign cannot create a candidate report." >&2
  exit 1
fi
report_args+=("${chosen_runs[@]}")
report_args+=(
  --baseline-tbt-p50-ms "$BASELINE_TBT_P50_MS"
  --baseline-failed-count "$BASELINE_FAILED_COUNT"
  --docker-compose submission/docker-compose.w4a16-gptq.yml
  --json-out "$campaign_dir/candidate_report.json"
  --md-out "$campaign_dir/candidate_report.md"
)
if [[ -n "$accuracy_diff" ]]; then
  report_args+=(--accuracy-diff-json "$accuracy_diff")
fi
if [[ -n "$output_diff" ]]; then
  report_args+=(--output-diff-json "$output_diff")
fi

python3 benchmark/candidate_report.py "${report_args[@]}" \
  > "$campaign_dir/candidate_report.stdout.md"

echo "Campaign complete: $campaign_dir"
echo "Candidate report: $campaign_dir/candidate_report.md"
