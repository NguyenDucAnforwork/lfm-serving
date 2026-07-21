#!/bin/bash
# Regenerate and validate the current decode-cost evidence package.
#
# This is intentionally local-only. It proves the artifacts and reports are
# internally consistent; it does not claim H200/Machete completion.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

BASELINE_RUN="${BASELINE_RUN:-results/fp8_h200shape_20260720-082416}"
PROFILE_RUN="${PROFILE_RUN:-results/fp8_profile_spec_local_20260720-083205}"
W4_RUN_A="${W4_RUN_A:-results/w4a16_gptq_marlin_no_conv_20260720-091148}"
W4_RUN_B="${W4_RUN_B:-results/w4a16_gptq_marlin_20260720-091153}"
W4_ARTIFACT="${W4_ARTIFACT:-artifacts/lfm2-w4a16-gptq-g128-no-conv}"
COMPOSE_FILE="${COMPOSE_FILE:-submission/docker-compose.w4a16-gptq.yml}"

AGG_JSON="${AGG_JSON:-results/w4a16_gptq_marlin_no_conv_local_aggregate.json}"
AGG_TXT="${AGG_TXT:-results/w4a16_gptq_marlin_no_conv_local_aggregate.txt}"
CANDIDATE_JSON="${CANDIDATE_JSON:-results/w4a16_gptq_marlin_no_conv_candidate_report.json}"
CANDIDATE_MD="${CANDIDATE_MD:-results/w4a16_gptq_marlin_no_conv_candidate_report.md}"
HANDOFF_JSON="${HANDOFF_JSON:-results/decode_cost_handoff_report.json}"
HANDOFF_MD="${HANDOFF_MD:-results/decode_cost_handoff_report.md}"

for path in \
  "$BASELINE_RUN/summary_ext.json" \
  "$PROFILE_RUN/profile_buckets_cuda.json" \
  "$PROFILE_RUN/kernel_next_step_cuda.json" \
  "$W4_RUN_A/summary_ext.json" \
  "$W4_RUN_B/summary_ext.json" \
  "$W4_ARTIFACT/config.json"; do
  [[ -f "$path" ]] || { echo "missing required evidence: $path" >&2; exit 1; }
done

"$PYTHON_BIN" -m py_compile \
  benchmark/analyze_profile_trace.py \
  benchmark/candidate_report.py \
  benchmark/compare_run_outputs.py \
  benchmark/decode_cost_handoff_report.py \
  benchmark/replay_trace.py \
  scripts/check_gpu_capability.py \
  scripts/quantize_w4a16_gptq.py

bash -n \
  scripts/run_decode_cost_campaign.sh \
  scripts/run_experiment.sh \
  scripts/run_profile_capture.sh \
  scripts/prepare_w4_submission.sh

"$PYTHON_BIN" scripts/validate_w4_candidate.py \
  --artifact "$W4_ARTIFACT" \
  --run-dir "$W4_RUN_B" \
  > /tmp/lfm_decode_cost_w4_validation.json

"$PYTHON_BIN" scripts/validate_submission_compose.py "$COMPOSE_FILE" \
  --candidate w4a16-gptq \
  --allow-placeholder-image \
  > /tmp/lfm_decode_cost_compose_validation.txt

"$PYTHON_BIN" benchmark/aggregate_runs.py "$W4_RUN_A" "$W4_RUN_B" \
  --baseline-tbt-p50-ms 4.0 \
  --baseline-failed-requests 4 \
  --json-out "$AGG_JSON" \
  > "$AGG_TXT"

"$PYTHON_BIN" benchmark/candidate_report.py \
  --candidate w4a16_gptq_marlin_no_conv \
  --runs "$W4_RUN_A" "$W4_RUN_B" \
  --baseline-tbt-p50-ms 4.0 \
  --baseline-failed-count 4 \
  --docker-compose "$COMPOSE_FILE" \
  --json-out "$CANDIDATE_JSON" \
  --md-out "$CANDIDATE_MD" \
  > /tmp/lfm_decode_cost_candidate_report_stdout.md

"$PYTHON_BIN" benchmark/decode_cost_handoff_report.py \
  --baseline-summary "$BASELINE_RUN/summary_ext.json" \
  --profile-buckets "$PROFILE_RUN/profile_buckets_cuda.json" \
  --kernel-next-step "$PROFILE_RUN/kernel_next_step_cuda.json" \
  --candidate-report "$CANDIDATE_JSON" \
  --aggregate "$AGG_JSON" \
  --md-out "$HANDOFF_MD" \
  --json-out "$HANDOFF_JSON"

# Verify output-diff guardrail: these historical runs intentionally did not
# capture output_text, so semantic diffing must fail loudly.
if "$PYTHON_BIN" benchmark/compare_run_outputs.py \
  --baseline "$BASELINE_RUN/results.jsonl" \
  --candidate "$W4_RUN_B/results.jsonl" \
  --candidate-modules attention_mlp_w4_g128_short_conv_lm_head_bf16 \
  --json-out /tmp/lfm_decode_cost_unexpected_output_diff.json \
  > /tmp/lfm_decode_cost_output_diff_stdout.txt 2> /tmp/lfm_decode_cost_output_diff_stderr.txt; then
  echo "compare_run_outputs unexpectedly succeeded without captured output_text" >&2
  exit 1
fi
grep -q "output_text missing" /tmp/lfm_decode_cost_output_diff_stderr.txt

"$PYTHON_BIN" - <<'PY' "$CANDIDATE_JSON" "$HANDOFF_JSON"
import json, sys
candidate = json.load(open(sys.argv[1]))
handoff = json.load(open(sys.argv[2]))
assert candidate["gates"]["candidate_passes"] is False
assert handoff["done_criteria"]["candidate_passes"] is False
assert handoff["done_criteria"]["no_additional_failures"] is True
assert handoff["done_criteria"]["decode_target_or_15pct_improvement"] is False
print("decode-cost evidence validation passed")
PY

echo "candidate report: $CANDIDATE_MD"
echo "handoff report: $HANDOFF_MD"
