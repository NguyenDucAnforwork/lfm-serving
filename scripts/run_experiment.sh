#!/bin/bash
# Run one controlled experiment: start the server with configs/<name>.env,
# wait for readiness, monitor VRAM (killing the server if it exceeds the 7GB
# hard limit), replay the full trace, stop the server, and record results
# under results/<run_id>/.
#
# Usage: scripts/run_experiment.sh <config_name> [extra replay_trace.py args]
#
# Optional env:
#   TRACE=trace_grading_spec_v2.jsonl WORKLOAD=spec MODEL_NAME=LFM2.5-1.2B-Instruct TOKENIZER_MODEL=LiquidAI/LFM2.5-1.2B-Instruct
#   KEEP_OUTPUT_TEXT=1  # persist generated text for exact output-diff analysis
#
# Defaults below are trace_grading_spec_v2.jsonl / WORKLOAD=spec (the current
# official workload, 420 scored requests, 0 warmup -- see gen_spec_trace.py).
# Override TRACE explicitly to replay the legacy trace_grading_public.jsonl.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_NAME="${1:?usage: run_experiment.sh <config_name> [extra args]}"
shift
CONFIG_FILE="$PROJECT_DIR/configs/${CONFIG_NAME}.env"
[[ -f "$CONFIG_FILE" ]] || { echo "missing config $CONFIG_FILE" >&2; exit 1; }

RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$PROJECT_DIR/results/$RUN_ID"
mkdir -p "$OUT_DIR"
cp "$CONFIG_FILE" "$OUT_DIR/config.env"

echo "=== Experiment: $RUN_ID ==="
cat "$CONFIG_FILE"

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a
export LOG_FILE="$OUT_DIR/server.log"
TRACE="${TRACE:-trace_grading_spec_v2.jsonl}"
WORKLOAD="${WORKLOAD:-spec}"
MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME:-$MODEL}}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$MODEL}"

if [[ "$WORKLOAD" == "spec" && "$TRACE" != *"spec"* ]]; then
  echo "ERROR: WORKLOAD=spec requires a spec trace (got TRACE=$TRACE)" >&2
  exit 2
fi

python3 - "$PROJECT_DIR/$TRACE" "$WORKLOAD" <<'PY'
import json
import sys

path, workload = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in open(path) if x.strip()]

if workload == "spec":
    assert len(rows) == 420, f"expected 420 rows in {path}, got {len(rows)}"
    assert sum(bool(r.get("in_warmup")) for r in rows) == 0, (
        f"{path} has warmup rows but WORKLOAD=spec scores all requests "
        "(official grader warmup_count=0) -- regenerate with gen_spec_trace.py"
    )
    assert {r["turn_idx"] for r in rows} == set(range(6)), f"unexpected turn_idx values in {path}"
    print(f"Trace validated: {path} -- 420 scored requests, 0 warmup")
else:
    print(f"Trace {path} loaded ({len(rows)} rows) -- WORKLOAD={workload}, skipping spec preflight")
PY
[[ $? -eq 0 ]] || { echo "TRACE PREFLIGHT FAILED" >&2; exit 2; }

bash "$SCRIPT_DIR/start_server.sh" &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/server.pid"
echo "server (api_server) pid: $SERVER_PID"

READY=0
for i in $(seq 1 90); do
  if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then READY=1; break; fi
  if grep -qiE "Traceback|CUDA out of memory" "$LOG_FILE" 2>/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  sleep 2
done

if [[ "$READY" != "1" ]]; then
  echo "SERVER FAILED TO START" | tee "$OUT_DIR/FAILED"
  tail -80 "$LOG_FILE"
  kill "$SERVER_PID" 2>/dev/null
  bash "$SCRIPT_DIR/stop_server.sh" "$OUT_DIR/server.pid"
  exit 1
fi

sleep 2
ENGINE_PID=$(grep -oE 'EngineCore pid=[0-9]+' "$LOG_FILE" 2>/dev/null \
  | sed -E 's/.*pid=([0-9]+)/\1/' | tail -1)
if [[ -n "$ENGINE_PID" ]] && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | awk '{gsub(/ /,""); print}' | grep -qx "$ENGINE_PID"; then
  ENGINE_PID=""
fi
if [[ -z "$ENGINE_PID" ]]; then
  ENGINE_PID=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
  | awk -F',' '/VLLM::EngineCore/{gsub(/ /,"",$1); print $1}' | tail -1)
fi
if [[ -z "$ENGINE_PID" ]]; then
  ENGINE_PID=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 > best) {best=$2+0; pid=$1}} END {if (pid) print pid}')
fi
echo "engine core pid: ${ENGINE_PID:-unknown}"

bash "$SCRIPT_DIR/vram_monitor.sh" "$ENGINE_PID" "$OUT_DIR/vram_samples.csv" "${VRAM_LIMIT_MIB:-7000}" "$SERVER_PID" &
MONITOR_PID=$!

RSS_MONITOR_PID=""
if [[ "${RSS_MONITOR:-0}" == "1" ]]; then
  bash "$SCRIPT_DIR/rss_monitor.sh" "$SERVER_PID" "$OUT_DIR/rss_samples.csv" 1 &
  RSS_MONITOR_PID=$!
  echo "rss monitor pid: $RSS_MONITOR_PID"
fi

# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/bin/activate"
REPLAY_EXTRA_ARGS=()
if [[ "${KEEP_OUTPUT_TEXT:-0}" == "1" ]]; then
  REPLAY_EXTRA_ARGS+=(--keep-output-text)
fi
python3 "$PROJECT_DIR/benchmark/replay_trace.py" \
  --trace "$PROJECT_DIR/$TRACE" \
  --workload "$WORKLOAD" \
  --model "$MODEL_NAME" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --name "$RUN_ID" --out-dir "$OUT_DIR" --verbose \
  "${REPLAY_EXTRA_ARGS[@]}" "$@" 2>&1 | tee "$OUT_DIR/replay.log"
REPLAY_STATUS=${PIPESTATUS[0]}

kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null
if [[ -n "$RSS_MONITOR_PID" ]]; then
  kill "$RSS_MONITOR_PID" 2>/dev/null
  wait "$RSS_MONITOR_PID" 2>/dev/null
fi
bash "$SCRIPT_DIR/stop_server.sh" "$OUT_DIR/server.pid" > "$OUT_DIR/stop.log" 2>&1
sleep 2

VRAM_PEAK_MIB=$(python3 -c "
import csv
peak = 0
try:
    with open('$OUT_DIR/vram_samples.csv') as f:
        for row in csv.reader(f):
            if row and row[0].strip().isdigit():
                peak = max(peak, int(row[0]))
except FileNotFoundError:
    pass
print(peak)
")
echo "$VRAM_PEAK_MIB" > "$OUT_DIR/vram_peak_mib.txt"
echo "VRAM peak: ${VRAM_PEAK_MIB} MiB"

# Contention check: did another tenant's process use GPU memory during this
# run? (column 5 = other_procs_mib; see vram_monitor.sh). We never act on
# this beyond reporting -- it's a confound flag for the results, not
# something we police.
OTHER_PEAK_MIB=$(python3 -c "
import csv
peak = 0
try:
    with open('$OUT_DIR/vram_samples.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[4].strip().lstrip('-').isdigit():
                peak = max(peak, int(row[4]))
except FileNotFoundError:
    pass
print(peak)
")
echo "$OTHER_PEAK_MIB" > "$OUT_DIR/other_tenant_peak_mib.txt"
if [[ "$OTHER_PEAK_MIB" -gt 100 ]]; then
  echo "WARNING: another GPU process used up to ${OTHER_PEAK_MIB} MiB during this run -- results may be contended, see other_tenant_peak_mib.txt" | tee "$OUT_DIR/CONTENDED"
fi
if [[ -f "$OUT_DIR/rss_samples.csv" ]]; then
  RSS_PEAK_KB=$(python3 -c "
import csv
peak = 0
try:
    with open('$OUT_DIR/rss_samples.csv') as f:
        for row in csv.reader(f):
            if row and row[0].strip().isdigit():
                peak = max(peak, int(row[0]))
except FileNotFoundError:
    pass
print(peak)
")
  echo "$RSS_PEAK_KB" > "$OUT_DIR/rss_peak_kb.txt"
  echo "RSS peak: ${RSS_PEAK_KB} KB ($(python3 -c "print(f'{${RSS_PEAK_KB}/1024/1024:.2f}')") GB)"
fi

echo "Results dir: $OUT_DIR"

if [[ -f "$OUT_DIR/results.jsonl" ]]; then
  python3 "$PROJECT_DIR/benchmark/summarize_run.py" "$OUT_DIR/results.jsonl" \
    > "$OUT_DIR/summary_ext.json" 2>/dev/null || true
  python3 "$PROJECT_DIR/benchmark/analyze_failures.py" "$OUT_DIR/results.jsonl" \
    --server-log "$OUT_DIR/server.log" > "$OUT_DIR/failure_analysis.txt" 2>/dev/null || true
fi

if [[ "${QUANTIZATION:-}" == "compressed-tensors" ]]; then
  MODEL_PATH="$MODEL"
  if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$PROJECT_DIR/$MODEL_PATH"
  fi
  python3 "$PROJECT_DIR/scripts/validate_w4_candidate.py" \
    --artifact "$MODEL_PATH" \
    --run-dir "$OUT_DIR" > "$OUT_DIR/w4_validation.json" 2>/dev/null || true
fi

exit "$REPLAY_STATUS"
