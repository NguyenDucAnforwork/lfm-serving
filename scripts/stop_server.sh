#!/bin/bash
# Stop any running vLLM server started by this project (dev/experiment use only).
set -uo pipefail

PIDFILE="${1:-/workspace/lfm-serving/logs/server.pid}"

if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE")
  if kill "$PID" 2>/dev/null; then
    echo "Sent SIGTERM to $PID"
  fi
fi

# vLLM's API server forks an EngineCore subprocess; make sure both die.
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && echo "Killed api_server process(es)" || true
sleep 1
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true

echo "Remaining GPU compute processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
