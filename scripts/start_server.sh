#!/bin/bash
# Start the vLLM OpenAI-compatible server for LiquidAI/LFM2.5-1.2B-Instruct.
#
# Used both for local dev/experiments (sources the project .venv, logs to a
# file, MODEL defaults to the HF repo id) and, unchanged, as the container
# submission entrypoint (no venv to source -- vllm/vllm-openai:v0.22.1
# already has vLLM on PATH; MODEL=/model; logs to stdout for `docker logs`).
#
# Usage:
#   scripts/start_server.sh [extra vllm args...]
#
# Config is driven by env vars so experiments can override one knob at a time:
#   MODEL, PORT, HOST, MAX_MODEL_LEN, GPU_MEM_UTIL, MAX_NUM_SEQS,
#   MAX_NUM_BATCHED_TOKENS, ENABLE_PREFIX_CACHING, ENABLE_CHUNKED_PREFILL,
#   PERFORMANCE_MODE, PREFIX_CACHE_HASH_ALGO, DISABLE_LOG_STATS, DISABLE_ACCESS_LOG,
#   QUANTIZATION, KV_CACHE_DTYPE, SPEC_METHOD, SPEC_TOKENS, SPEC_MODEL,
#   SPECULATIVE_CONFIG, COMPILATION_CONFIG, ENFORCE_EAGER, MAX_NUM_PARTIAL_PREFILLS,
#   BLOCK_SIZE, LINEAR_BACKEND, PROFILER_CONFIG, CPU_PIN, EXTRA_VLLM_ARGS,
#   VLLM_VENV, LOG_FILE
#
# The QUANTIZATION/KV_CACHE_DTYPE/SPEC_*/COMPILATION_CONFIG/MAX_NUM_PARTIAL_PREFILLS/
# BLOCK_SIZE/CPU_PIN knobs are under evaluation this session (see next_step.md T1-T8);
# all default to empty/off (no behavior change) so they never silently alter the
# already-validated best.env / H200 profile behavior.
#
# Defaults below are the winning config from EXPERIMENTS.md (ERS 55.2% on the
# public trace, ~5.6GiB VRAM): max_model_len=4608, gpu_mem_util=0.22,
# max_num_seqs=4, max_num_batched_tokens=256, prefix caching on. The
# PERFORMANCE_MODE/PREFIX_CACHE_HASH_ALGO/DISABLE_* knobs default to empty/off
# (vLLM's own defaults) -- they are additional low-risk candidates under
# evaluation (see EXPERIMENTS.md "Additional vLLM tuning"), not yet proven
# wins, so they must be opted into explicitly rather than silently changing
# already-validated behavior.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="${MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"   # empty = vllm default (uses MODEL itself, e.g. the literal path "/model" -- wrong for the submission API's advertised model name, so the H200 profile sets this explicitly)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4608}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.22}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-}"   # empty = vllm default (forced on anyway w/ prefix caching, see EXPERIMENTS.md)
PERFORMANCE_MODE="${PERFORMANCE_MODE:-}"               # empty = vllm default (balanced); or interactivity/throughput
PREFIX_CACHE_HASH_ALGO="${PREFIX_CACHE_HASH_ALGO:-}"   # empty = vllm default (sha256); or xxhash/xxhash_cbor/sha256_cbor
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-0}"            # 1 = pass --disable-log-stats (CPU/logging overhead reduction)
DISABLE_ACCESS_LOG="${DISABLE_ACCESS_LOG:-0}"          # 1 = pass --disable-uvicorn-access-log
QUANTIZATION="${QUANTIZATION:-}"                       # empty = none (bf16); or fp8, torchao, bitsandbytes, ... (T1/T2)
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"                    # empty = vllm default (auto); or fp8, fp8_e5m2, ... (T3)
SPEC_METHOD="${SPEC_METHOD:-}"                          # empty = no speculative decoding; or ngram, ngram_gpu, suffix (T6)
SPEC_TOKENS="${SPEC_TOKENS:-}"                          # num speculative tokens per step, used with SPEC_METHOD
SPEC_MODEL="${SPEC_MODEL:-}"                            # draft model path, only for methods that need one
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"            # raw JSON, takes precedence over SPEC_METHOD/SPEC_TOKENS/SPEC_MODEL if set
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"            # raw JSON passed to --compilation-config (T4)
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"                     # 1 = pass --enforce-eager (T4 diagnostic: measures CUDA-graph benefit)
MAX_NUM_PARTIAL_PREFILLS="${MAX_NUM_PARTIAL_PREFILLS:-}" # empty = vllm default (1) (T7)
BLOCK_SIZE="${BLOCK_SIZE:-}"                            # empty = vllm default (T7)
LINEAR_BACKEND="${LINEAR_BACKEND:-}"                    # empty = vLLM auto; or machete/marlin/triton/etc. for backend-forcing experiments
PROFILER_CONFIG="${PROFILER_CONFIG:-}"                  # raw JSON passed to --profiler-config
CPU_PIN="${CPU_PIN:-}"                                  # e.g. "0-2" = exec under `taskset -c 0-2` (emulates the 3-CPU-core grading env)
VLLM_VENV="${VLLM_VENV:-$PROJECT_DIR/.venv}"
LOG_FILE="${LOG_FILE:-}"   # empty = log to stdout (container-friendly); set to a path for local dev
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# Keep this project's footprint isolated from other GPU users on a shared box.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Only local dev has a project venv to activate; the submission container
# already has vLLM installed system-wide.
if [[ -f "$VLLM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VLLM_VENV/bin/activate"
fi

ARGS=(
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --dtype bfloat16
)

if [[ -n "$SERVED_MODEL_NAME" ]]; then
  ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi

if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  ARGS+=(--enable-chunked-prefill)
elif [[ "$ENABLE_CHUNKED_PREFILL" == "0" ]]; then
  ARGS+=(--no-enable-chunked-prefill)
fi

if [[ -n "$PERFORMANCE_MODE" ]]; then
  ARGS+=(--performance-mode "$PERFORMANCE_MODE")
fi

if [[ -n "$PREFIX_CACHE_HASH_ALGO" ]]; then
  ARGS+=(--prefix-caching-hash-algo "$PREFIX_CACHE_HASH_ALGO")
fi

if [[ "$DISABLE_LOG_STATS" == "1" ]]; then
  ARGS+=(--disable-log-stats)
fi

if [[ "$DISABLE_ACCESS_LOG" == "1" ]]; then
  ARGS+=(--disable-uvicorn-access-log)
fi

if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ -n "$KV_CACHE_DTYPE" ]]; then
  ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if [[ -n "$SPECULATIVE_CONFIG" ]]; then
  ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
elif [[ -n "$SPEC_METHOD" ]]; then
  ARGS+=(--spec-method "$SPEC_METHOD")
  [[ -n "$SPEC_TOKENS" ]] && ARGS+=(--spec-tokens "$SPEC_TOKENS")
  [[ -n "$SPEC_MODEL" ]] && ARGS+=(--spec-model "$SPEC_MODEL")
fi

if [[ -n "$COMPILATION_CONFIG" ]]; then
  ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ -n "$MAX_NUM_PARTIAL_PREFILLS" ]]; then
  ARGS+=(--max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS")
fi

if [[ -n "$BLOCK_SIZE" ]]; then
  ARGS+=(--block-size "$BLOCK_SIZE")
fi

if [[ -n "$LINEAR_BACKEND" ]]; then
  ARGS+=(--linear-backend "$LINEAR_BACKEND")
fi

if [[ -n "$PROFILER_CONFIG" ]]; then
  ARGS+=(--profiler-config "$PROFILER_CONFIG")
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_VLLM_ARGS)
  ARGS+=("${EXTRA_ARR[@]}")
fi

echo "Launching vLLM server:"
echo "  model=$MODEL host=$HOST port=$PORT max_model_len=$MAX_MODEL_LEN gpu_mem_util=$GPU_MEM_UTIL"
echo "  max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-<default>}"
echo "  prefix_caching=$ENABLE_PREFIX_CACHING chunked_prefill=${ENABLE_CHUNKED_PREFILL:-<default>}"
echo "  performance_mode=${PERFORMANCE_MODE:-<default>} hash_algo=${PREFIX_CACHE_HASH_ALGO:-<default>}"
echo "  disable_log_stats=$DISABLE_LOG_STATS disable_access_log=$DISABLE_ACCESS_LOG"
echo "  quantization=${QUANTIZATION:-<none>} kv_cache_dtype=${KV_CACHE_DTYPE:-<default>}"
echo "  spec_method=${SPEC_METHOD:-<none>} spec_tokens=${SPEC_TOKENS:-} speculative_config=${SPECULATIVE_CONFIG:-<none>}"
echo "  compilation_config=${COMPILATION_CONFIG:-<default>} enforce_eager=$ENFORCE_EAGER"
echo "  max_num_partial_prefills=${MAX_NUM_PARTIAL_PREFILLS:-<default>} block_size=${BLOCK_SIZE:-<default>}"
echo "  linear_backend=${LINEAR_BACKEND:-<auto>} profiler_config=${PROFILER_CONFIG:-<none>}"
echo "  cpu_pin=${CPU_PIN:-<none>} extra_args=${EXTRA_VLLM_ARGS:-<none>}"
echo "  log=${LOG_FILE:-<stdout>}"

RUN_CMD=(python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" "$@")
if [[ -n "$CPU_PIN" ]]; then
  RUN_CMD=(taskset -c "$CPU_PIN" "${RUN_CMD[@]}")
fi

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  exec "${RUN_CMD[@]}" > "$LOG_FILE" 2>&1
else
  exec "${RUN_CMD[@]}"
fi
