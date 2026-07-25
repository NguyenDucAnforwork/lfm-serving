#!/usr/bin/env bash
# Poll a running vLLM server's /metrics endpoint once per second, appending
# timestamped snapshots of the counters relevant to diagnosing scheduler/
# cache behavior (queue depth, KV usage, prefix-cache hits, preemptions).
#
# Usage: scrape_metrics.sh [out_file] [metrics_url]
set -euo pipefail

OUT="${1:-metrics.log}"
URL="${2:-http://127.0.0.1:8000/metrics}"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

while true; do
  printf '\n# timestamp=%s\n' "$(date +%s.%N)" >> "$OUT"
  curl -fsS "$URL" 2>/dev/null | grep -E \
    'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|prefix_cache_queries_total|prefix_cache_hits_total|num_preemptions_total|prompt_tokens_cached_total|request_queue_time_seconds_(sum|count))' \
    >> "$OUT" || true
  sleep 1
done
