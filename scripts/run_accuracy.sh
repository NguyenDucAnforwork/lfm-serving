#!/bin/bash
# Run GPQA accuracy against a server started from configs/<name>.env.
#
# This is for experiment validation only. It starts/stops vLLM like
# run_experiment.sh, writes per-sample JSONL, stdout, and summary JSON under
# results/<run_id>/accuracy/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_NAME="${1:?usage: run_accuracy.sh <config_name> [run_suffix] [extra gpqa_eval args...]}"
RUN_SUFFIX="${2:-accuracy}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

CONFIG_FILE="$PROJECT_DIR/configs/${CONFIG_NAME}.env"
[[ -f "$CONFIG_FILE" ]] || { echo "missing config $CONFIG_FILE" >&2; exit 1; }

RUN_ID="${CONFIG_NAME}_${RUN_SUFFIX}_$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$PROJECT_DIR/results/$RUN_ID"
ACC_DIR="$OUT_DIR/accuracy"
mkdir -p "$ACC_DIR"
cp "$CONFIG_FILE" "$OUT_DIR/config.env"

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

export LOG_FILE="$OUT_DIR/server.log"
MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME:-$MODEL}}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-8000}}"

bash "$SCRIPT_DIR/start_server.sh" &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/server.pid"
trap 'bash "$SCRIPT_DIR/stop_server.sh" "$OUT_DIR/server.pid" >/dev/null 2>&1 || true' EXIT

READY=0
for _ in $(seq 1 180); do
  if curl -fsS "$BASE_URL/health" >/dev/null; then
    READY=1
    break
  fi
  if grep -qiE "Traceback|CUDA out of memory" "$LOG_FILE" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "SERVER FAILED TO START" | tee "$OUT_DIR/FAILED"
  tail -80 "$LOG_FILE" || true
  exit 1
fi

if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
fi

python3 "$PROJECT_DIR/benchmark/gpqa_eval.py" \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --output "$ACC_DIR/gpqa.jsonl" \
  --verbose "$@" 2>&1 | tee "$ACC_DIR/gpqa_stdout.txt"

python3 "$PROJECT_DIR/benchmark/summarize_accuracy.py" "$ACC_DIR/gpqa.jsonl" \
  --json-out "$ACC_DIR/gpqa_summary.json" > "$ACC_DIR/gpqa_summary_stdout.json"

if [[ "${QUANTIZATION:-}" == "compressed-tensors" ]]; then
  MODEL_PATH="$MODEL"
  if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$PROJECT_DIR/$MODEL_PATH"
  fi
  python3 "$PROJECT_DIR/scripts/validate_w4_candidate.py" \
    --artifact "$MODEL_PATH" \
    --run-dir "$OUT_DIR" > "$OUT_DIR/w4_validation.json" 2>/dev/null || true
fi

echo "Accuracy results: $ACC_DIR"
