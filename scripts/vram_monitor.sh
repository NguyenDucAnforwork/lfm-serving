#!/bin/bash
# Poll nvidia-smi for one process's VRAM usage and log CSV samples.
# If usage exceeds the limit, kill the offending process (and, if given, the
# parent server PID too) -- this is the hard 7GB safety valve.
#
# Also logs total GPU memory/utilization each sample so post-hoc analysis can
# tell whether OTHER tenants' processes were active during the run on this
# shared box (a real confound for TTFT/TPOT measurements -- see EXPERIMENTS.md
# "GPU contention" note). We never touch other processes, only observe them.
#
# CSV columns: our_pid_used_mib,timestamp,total_gpu_used_mib,gpu_util_pct,other_procs_mib
#
# Usage: vram_monitor.sh <engine_pid> <out_csv> [limit_mib] [server_pid]
set -uo pipefail

PID="${1:-}"
OUT="${2:?output csv path required}"
LIMIT="${3:-7000}"
KILL_TARGET="${4:-}"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

while true; do
  USED=""
  if [[ -n "$PID" ]]; then
    USED=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' -v p="$PID" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==p) print $2}')
  fi
  TS=$(date +%s.%N)

  GPU_STATS=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  TOTAL_USED=$(echo "$GPU_STATS" | awk -F',' '{gsub(/ /,"",$1); print $1}')
  GPU_UTIL=$(echo "$GPU_STATS" | awk -F',' '{gsub(/ /,"",$2); print $2}')
  OTHER_MIB=""
  if [[ -n "$TOTAL_USED" && "$TOTAL_USED" =~ ^[0-9]+$ ]]; then
    OTHER_MIB=$(( TOTAL_USED - ${USED:-0} ))
  fi

  echo "${USED:-0},${TS},${TOTAL_USED:-0},${GPU_UTIL:-0},${OTHER_MIB:-0}" >> "$OUT"

  if [[ -n "$USED" ]] && [[ "$USED" =~ ^[0-9]+$ ]] && (( USED > LIMIT )); then
    echo "[vram_monitor] LIMIT EXCEEDED: ${USED} MiB > ${LIMIT} MiB — killing server (pid=${PID}, target=${KILL_TARGET})" >&2
    [[ -n "$KILL_TARGET" ]] && kill -9 "$KILL_TARGET" 2>/dev/null
    [[ -n "$PID" ]] && kill -9 "$PID" 2>/dev/null
    exit 1
  fi
  sleep 1
done
