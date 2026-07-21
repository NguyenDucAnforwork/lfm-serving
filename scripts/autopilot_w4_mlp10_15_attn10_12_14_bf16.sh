#!/usr/bin/env bash
# Continue the current W4 G64 mixed-precision experiment unattended.
# - Polls export/disk progress every 5s.
# - Validates the artifact after export.
# - Runs trace.
# - Runs accuracy gate only if trace has 0 failures and TBT <= 3.5 ms.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

ARTIFACT="/dev/shm/artifacts-work/lfm2-w4a16-gptq-g64-mlp10-15-attn10-12-14-bf16"
EXPORT_LOG="${EXPORT_LOG:-$(ls -t results/export_logs/w4_g64_mlp10_15_attn10_12_14_bf16_*.log 2>/dev/null | head -1)}"
TRACE_CONFIG="w4a16_gptq_marlin_g64_mlp10_15_attn10_12_14_bf16_autodetect_trace"
ACC_CONFIG="w4a16_gptq_marlin_g64_mlp10_15_attn10_12_14_bf16_autodetect"
AUTO_LOG="results/autopilot_w4_mlp10_15_attn10_12_14_bf16.log"
STATUS_JSON="results/autopilot_w4_mlp10_15_attn10_12_14_bf16.status.json"
ACCURACY_ROOT="results/accuracy_w4_g64_mlp10_15_attn10_12_14_bf16_gate"

mkdir -p results/export_logs results

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$AUTO_LOG"
}

write_status() {
  local stage="$1"
  local detail="$2"
  python3 - "$STATUS_JSON" "$stage" "$detail" <<'PY'
import json, sys, time
path, stage, detail = sys.argv[1:4]
with open(path, "w") as f:
    json.dump({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "detail": detail}, f, indent=2)
PY
}

latest_progress() {
  if [[ -n "${EXPORT_LOG:-}" && -f "$EXPORT_LOG" ]]; then
    tail -c 2500 "$EXPORT_LOG" | tr '\r' '\n' | tail -5 | sed 's/^/export: /' | tee -a "$AUTO_LOG" >/dev/null
  fi
  df -h /workspace /dev/shm | tee -a "$AUTO_LOG" >/dev/null
  du -sh /dev/shm/artifacts-work/* 2>/dev/null | sort -h | tee -a "$AUTO_LOG" >/dev/null || true
}

export_still_running() {
  pgrep -f "scripts/quantize_w4a16_gptq.py.*lfm2-w4a16-gptq-g64-mlp10-15-attn10-12-14-bf16" >/dev/null
}

wait_for_export() {
  write_status "export" "waiting for current quantize_w4a16_gptq.py process"
  log "Waiting for export. log=${EXPORT_LOG:-missing}"
  while export_still_running; do
    latest_progress
    sleep 5
  done
  latest_progress
  if [[ ! -d "$ARTIFACT" ]]; then
    log "Export process ended but artifact directory is missing: $ARTIFACT"
    write_status "blocked" "artifact directory missing after export"
    exit 1
  fi
  if [[ -n "${EXPORT_LOG:-}" && -f "$EXPORT_LOG" ]] && grep -qiE "Traceback|No space left|CUDA out of memory|Killed" "$EXPORT_LOG"; then
    log "Export log contains failure markers."
    write_status "blocked" "export log has failure marker"
    exit 1
  fi
  log "Export completed; artifact present at $ARTIFACT"
}

validate_artifact() {
  write_status "validate" "validating W4 artifact and expected ignores"
  log "Validating artifact"
  python3 scripts/validate_w4_candidate.py --artifact "$ARTIFACT" --expected-group-size 64 \
    > results/w4_g64_mlp10_15_attn10_12_14_bf16_validation.json
  python3 - "$ARTIFACT/config.json" <<'PY' | tee results/w4_g64_mlp10_15_attn10_12_14_bf16_ignore_check.txt
import json, sys
d=json.load(open(sys.argv[1]))
ign=set(d.get("quantization_config",{}).get("ignore",[]))
checks=[
 "lm_head",
 "model.layers.10.feed_forward.w2",
 "model.layers.15.feed_forward.w3",
 "model.layers.10.self_attn.q_proj",
 "model.layers.12.self_attn.v_proj",
 "model.layers.14.self_attn.out_proj",
]
for c in checks:
    print(f"{c}: {c in ign}")
PY
  log "Validation done"
}

run_trace() {
  write_status "trace" "running 420-request trace"
  log "Running trace config: $TRACE_CONFIG"
  set +e
  KEEP_OUTPUT_TEXT=1 bash scripts/run_experiment.sh "$TRACE_CONFIG" 2>&1 \
    | tee "results/export_logs/w4_g64_mlp10_15_attn10_12_14_bf16_trace_$(date -u +%Y%m%d-%H%M%S).log"
  local status=${PIPESTATUS[0]}
  set -e
  local latest_run
  latest_run="$(ls -dt results/${TRACE_CONFIG}_* 2>/dev/null | head -1)"
  if [[ "$status" -ne 0 || -z "$latest_run" ]]; then
    log "Trace failed to complete. status=$status latest_run=${latest_run:-missing}"
    write_status "blocked" "trace command failed"
    exit 1
  fi
  log "Trace results: $latest_run"
  set +e
  python3 - "$latest_run/summary_ext.json" "$latest_run/server.log" <<'PY' | tee "results/w4_g64_mlp10_15_attn10_12_14_bf16_trace_gate.txt"
import json, sys
summary=json.load(open(sys.argv[1]))
server_log=open(sys.argv[2], errors="ignore").read()
n_errors=summary.get("n_errors", 999)
failed=summary.get("failed_requests", 999)
tbt=summary.get("tpot_ms_mean", 999)
marlin=("MarlinLinearKernel" in server_log) or ("MacheteLinearKernel" in server_log)
autodetect=("quantization=compressed-tensors" in server_log)
ok=(n_errors == 0 and failed == 0 and tbt <= 3.5 and marlin and autodetect)
print(json.dumps({
  "trace_dir": str(sys.argv[1]).rsplit("/", 1)[0],
  "n_errors": n_errors,
  "failed_requests": failed,
  "tbt_ms_mean": tbt,
  "tbt_ms_p50": summary.get("tpot_ms_p50"),
  "tbt_ms_p95": summary.get("tpot_ms_p95"),
  "ttft_ms_mean": summary.get("ttft_ms_mean"),
  "ers_pct": summary.get("ers_mean_pct"),
  "marlin_or_machete_log_seen": marlin,
  "compressed_tensors_autodetect_seen": autodetect,
  "pass_for_accuracy": ok,
}, indent=2))
sys.exit(0 if ok else 2)
PY
  local gate_status=${PIPESTATUS[0]}
  set -e
  if [[ "$gate_status" -ne 0 ]]; then
    log "Trace completed but failed gate; accuracy will be skipped."
    write_status "rejected" "trace completed but failed speed/failure/backend gate"
    return 0
  fi
}

run_accuracy_if_trace_passed() {
  if ! grep -q '"pass_for_accuracy": true' results/w4_g64_mlp10_15_attn10_12_14_bf16_trace_gate.txt; then
    log "Trace did not pass speed/failure/backend gate; skipping accuracy."
    write_status "rejected" "trace gate failed; accuracy skipped"
    exit 0
  fi
  write_status "accuracy" "running ARC full + GSM8K 200 + GPQA if token exists"
  log "Trace gate passed; running accuracy gate."
  mkdir -p "$ACCURACY_ROOT"
  if [[ ! -d "$ACCURACY_ROOT/bf16_h200shape" && -d results/accuracy_w4_g64_mlp10_15_bf16_gate/bf16_h200shape ]]; then
    cp -a results/accuracy_w4_g64_mlp10_15_bf16_gate/bf16_h200shape "$ACCURACY_ROOT/bf16_h200shape"
  fi
  if [[ ! -d "$ACCURACY_ROOT/fp8_h200shape" && -d results/accuracy_w4_gate/fp8_h200shape ]]; then
    cp -a results/accuracy_w4_gate/fp8_h200shape "$ACCURACY_ROOT/fp8_h200shape"
  fi
  OUT_ROOT="$ACCURACY_ROOT" \
  RUN_CONFIGS="$ACC_CONFIG" \
  TOKENIZER_MODEL="$ARTIFACT" \
  NUM_CONCURRENT=8 \
  MAX_LENGTH=4096 \
  bash scripts/run_w4_accuracy_gate.sh 2>&1 \
    | tee "results/export_logs/w4_g64_mlp10_15_attn10_12_14_bf16_accuracy_$(date -u +%Y%m%d-%H%M%S).log"
  python3 benchmark/make_accuracy_gate.py \
    --results-root "$ACCURACY_ROOT" \
    --baseline-config bf16_h200shape \
    --candidate-config "$ACC_CONFIG" \
    --json-out "$ACCURACY_ROOT/accuracy_gate.mlp10_15_attn10_12_14_bf16_autodetect.draft.json"
  write_status "done" "accuracy gate draft written"
  log "Accuracy gate draft written: $ACCURACY_ROOT/accuracy_gate.mlp10_15_attn10_12_14_bf16_autodetect.draft.json"
}

wait_for_export
validate_artifact
run_trace
run_accuracy_if_trace_passed
