#!/bin/bash
# Capture a short torch CPU/CUDA profile around a trace replay.
#
# Intended for a CUDA-13-capable H200/official-like host. This host cannot run
# it if the installed torch/vLLM CUDA stack cannot initialize the driver.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_NAME="${1:-fp8_h200shape}"
RUN_NAME="${2:-${CONFIG_NAME}_profile_$(date +%Y%m%d-%H%M%S)}"
PROFILE_DIR="${PROFILE_DIR:-$PROJECT_DIR/results/$RUN_NAME/torch_profile}"
TRACE="${TRACE:-trace_grading_spec.jsonl}"
WORKLOAD="${WORKLOAD:-spec}"
MODEL_NAME="${MODEL_NAME:-LFM2.5-1.2B-Instruct}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"

mkdir -p "results/$RUN_NAME" "$PROFILE_DIR"

set -a
# shellcheck disable=SC1090
source "$PROJECT_DIR/configs/${CONFIG_NAME}.env"
set +a

export LOG_FILE="$PROJECT_DIR/results/$RUN_NAME/server.log"
export PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"ignore_frontend\":true,\"wait_iterations\":5,\"warmup_iterations\":2,\"active_iterations\":12,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_with_flops\":false}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-8000}}"

bash scripts/start_server.sh &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if curl -fsS "$BASE_URL/health" >/dev/null; then
    READY=1
    break
  fi
  sleep 1
done
if [[ "$READY" != "1" ]]; then
  echo "Server did not become healthy at $BASE_URL within 180s; see $LOG_FILE" >&2
  exit 1
fi

curl -fsS -X POST "$BASE_URL/start_profile"
python benchmark/replay_trace.py \
  --trace "$TRACE" \
  --workload "$WORKLOAD" \
  --name "$RUN_NAME" \
  --out-dir "results/$RUN_NAME" \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --verbose
curl -fsS -X POST "$BASE_URL/stop_profile"

python benchmark/summarize_run.py "results/$RUN_NAME/results.jsonl" \
  > "results/$RUN_NAME/summary_ext.json"
python benchmark/analyze_failures.py "results/$RUN_NAME/results.jsonl" \
  --server-log "results/$RUN_NAME/server.log" \
  > "results/$RUN_NAME/failure_analysis.txt"
python benchmark/analyze_profile_trace.py "results/$RUN_NAME/torch_profile" \
  --json-out "results/$RUN_NAME/profile_buckets.json" \
  > "results/$RUN_NAME/profile_buckets.txt"

echo "Profile: results/$RUN_NAME/torch_profile"
echo "Summary: results/$RUN_NAME/summary_ext.json"
echo "Profile buckets: results/$RUN_NAME/profile_buckets.txt"
