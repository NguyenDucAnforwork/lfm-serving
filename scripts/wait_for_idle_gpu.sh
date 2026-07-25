#!/bin/bash
# Wait until no OTHER tenant's process is using the GPU, so our benchmark
# runs are measured without external contention (this is a shared box --
# see EXPERIMENTS.md "GPU contention" note). We never kill or touch other
# processes; we just wait for them to finish on their own, up to a timeout.
#
# Usage: wait_for_idle_gpu.sh [timeout_s] [poll_interval_s]
set -uo pipefail

TIMEOUT="${1:-600}"
INTERVAL="${2:-5}"
ELAPSED=0

while true; do
  OTHER_MIB=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s+=$1} END {print s+0}')
  if [[ "${OTHER_MIB:-0}" -le 50 ]]; then
    echo "[wait_for_idle_gpu] GPU idle (${OTHER_MIB:-0} MiB in use by any process) after ${ELAPSED}s"
    exit 0
  fi
  if (( ELAPSED >= TIMEOUT )); then
    echo "[wait_for_idle_gpu] TIMEOUT after ${TIMEOUT}s -- GPU still has ${OTHER_MIB} MiB in use by other process(es); proceeding anyway (results may be contended)" >&2
    exit 1
  fi
  sleep "$INTERVAL"
  ELAPSED=$(( ELAPSED + INTERVAL ))
done
