#!/bin/bash
# Poll total RSS (VmRSS) of a process tree once per second, log CSV samples.
# Used to check the submission stays under the grading environment's 8GB RAM
# limit (see next_step.md T5/Block B). Sums VmRSS across the root PID and all
# its descendants (vLLM's API server forks an EngineCore subprocess).
#
# Usage: rss_monitor.sh <root_pid> <out_csv> [interval_s]
set -uo pipefail

ROOT_PID="${1:?root pid required}"
OUT="${2:?output csv path required}"
INTERVAL="${3:-1}"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

descendants() {
  local pid="$1"
  local children
  children=$(pgrep -P "$pid" 2>/dev/null)
  echo "$pid"
  for c in $children; do
    descendants "$c"
  done
}

while true; do
  if ! kill -0 "$ROOT_PID" 2>/dev/null; then
    echo "[rss_monitor] root pid $ROOT_PID gone, exiting" >&2
    exit 0
  fi
  TOTAL_KB=0
  for pid in $(descendants "$ROOT_PID"); do
    if [[ -r "/proc/$pid/status" ]]; then
      KB=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
      if [[ -n "$KB" ]]; then
        TOTAL_KB=$(( TOTAL_KB + KB ))
      fi
    fi
  done
  TS=$(date +%s.%N)
  echo "${TOTAL_KB},${TS}" >> "$OUT"
  sleep "$INTERVAL"
done
