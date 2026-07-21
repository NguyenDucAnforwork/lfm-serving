#!/bin/bash
# Run the W4 pre-submit accuracy gate:
#   - BF16 h200-shape reference
#   - FP8 h200-shape reference
#   - W4A16 GPTQ no-conv candidate
# across ARC Challenge full, GSM8K limit 200, and GPQA Diamond official when
# HF_TOKEN is available or REQUIRE_GPQA=1.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

OUT_ROOT="${OUT_ROOT:-results/accuracy_w4_gate}"
EVAL_VENV="${EVAL_VENV:-.venv-eval}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_NAME="${MODEL_NAME:-LFM2.5-1.2B-Instruct}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
REQUIRE_GPQA="${REQUIRE_GPQA:-0}"
RUN_CONFIGS="${RUN_CONFIGS:-bf16_h200shape fp8_h200shape w4a16_gptq_marlin}"
CURRENT_CONFIG=""

cleanup_current() {
  if [[ -n "$CURRENT_CONFIG" ]]; then
    bash scripts/stop_server.sh "$OUT_ROOT/${CURRENT_CONFIG}_server.pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup_current EXIT

mkdir -p "$OUT_ROOT"

if [[ ! -x "$EVAL_VENV/bin/lm_eval" ]]; then
  python3 -m venv "$EVAL_VENV"
  "$EVAL_VENV/bin/pip" install --upgrade pip
  "$EVAL_VENV/bin/pip" install "lm-eval[api]==0.4.12" transformers datasets
fi

start_config() {
  local config_name="$1"
  local config_file="$PROJECT_DIR/configs/${config_name}.env"
  [[ -f "$config_file" ]] || { echo "missing config $config_file" >&2; exit 1; }

  set -a
  # shellcheck disable=SC1090
  source "$config_file"
  set +a

  export LOG_FILE="$OUT_ROOT/${config_name}_server.log"
  bash scripts/start_server.sh &
  SERVER_PID=$!
  CURRENT_CONFIG="$config_name"
  echo "$SERVER_PID" > "$OUT_ROOT/${config_name}_server.pid"

  local ready=0
  for _ in $(seq 1 240); do
    if curl -fsS "$BASE_URL/health" >/dev/null; then
      ready=1
      break
    fi
    if grep -qiE "Traceback|CUDA out of memory|KeyError|MacheteLinearKernel requires" "$LOG_FILE" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if [[ "$ready" != "1" ]]; then
    echo "server failed for $config_name" >&2
    tail -100 "$LOG_FILE" >&2 || true
    exit 1
  fi
}

stop_config() {
  local config_name="$1"
  bash scripts/stop_server.sh "$OUT_ROOT/${config_name}_server.pid" >/dev/null 2>&1 || true
  CURRENT_CONFIG=""
  sleep 2
}

run_lm_eval_for_config() {
  local config_name="$1"
  local out_dir="$OUT_ROOT/$config_name"
  mkdir -p "$out_dir"

  "$EVAL_VENV/bin/lm_eval" \
    --model local-completions \
    --model_args "model=${MODEL_NAME},tokenizer=${TOKENIZER_MODEL},base_url=${BASE_URL}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_length=${MAX_LENGTH}" \
    --tasks arc_challenge \
    --batch_size 1 \
    --seed 1234 \
    --output_path "$out_dir/arc_challenge" \
    --log_samples 2>&1 | tee "$out_dir/arc_challenge.log"

  "$EVAL_VENV/bin/lm_eval" \
    --model local-chat-completions \
    --model_args "model=${MODEL_NAME},tokenizer=${TOKENIZER_MODEL},base_url=${BASE_URL}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT}" \
    --tasks gsm8k \
    --limit 200 \
    --batch_size 1 \
    --apply_chat_template \
    --seed 1234 \
    --output_path "$out_dir/gsm8k" \
    --log_samples 2>&1 | tee "$out_dir/gsm8k.log"

  if [[ -n "${HF_TOKEN:-}" || "$REQUIRE_GPQA" == "1" ]]; then
    "$EVAL_VENV/bin/lm_eval" \
      --model local-completions \
      --model_args "model=${MODEL_NAME},tokenizer=${TOKENIZER_MODEL},base_url=${BASE_URL}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_length=${MAX_LENGTH}" \
      --tasks gpqa_diamond_zeroshot \
      --batch_size 1 \
      --seed 1234 \
      --output_path "$out_dir/gpqa_diamond_zeroshot" \
      --log_samples 2>&1 | tee "$out_dir/gpqa_diamond_zeroshot.log"
  else
    echo "Skipping official GPQA: HF_TOKEN not present and REQUIRE_GPQA=0" | tee "$out_dir/gpqa_diamond_zeroshot.SKIPPED"
  fi
}

for config_name in $RUN_CONFIGS; do
  echo "=== accuracy config: $config_name ==="
  start_config "$config_name"
  run_lm_eval_for_config "$config_name"
  stop_config "$config_name"
done

python3 benchmark/make_accuracy_gate.py \
  --results-root "$OUT_ROOT" \
  --json-out "$OUT_ROOT/accuracy_gate.draft.json" \
  $( [[ "$REQUIRE_GPQA" == "1" ]] && echo --require-gpqa )

echo "Draft gate written to $OUT_ROOT/accuracy_gate.draft.json"
echo "Manual output coherence/format inspection is still required before setting pass=true."
