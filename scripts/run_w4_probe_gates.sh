#!/bin/bash
# Mandatory pre-submit gates for a W4A16 GPTQ H200 backend probe.
#
# This script is stricter than a local smoke test and intentionally does not
# decide that W4 is a winning candidate. It only answers: "is this compose/image
# safe enough to spend one official H200 submission slot as a backend probe?"
#
# Required before running:
#   1. Stage the no-conv W4 artifact into submission/model, e.g. with
#      scripts/prepare_w4_submission.sh --probe after the accuracy gate passes.
#   2. Replace the placeholder image in submission/docker-compose.w4a16-gptq.yml.
#   3. Provide precomputed accuracy gate evidence via ACCURACY_GATE_JSON.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-submission/docker-compose.w4a16-gptq.yml}"
ARTIFACT_DIR="submission/model"
TRACE="${TRACE:-trace_grading_spec.jsonl}"
WORKLOAD="${WORKLOAD:-spec}"
MODEL_NAME="${MODEL_NAME:-LFM2.5-1.2B-Instruct}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-LiquidAI/LFM2.5-1.2B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_ACCURACY="${RUN_ACCURACY:-0}"
ACCURACY_GATE_JSON="${ACCURACY_GATE_JSON:-}"
IMAGE_TAG="${IMAGE_TAG:-lfm-serving:w4a16-gptq-probe-local}"
OUT_DIR="${OUT_DIR:-results/w4_probe_gates_$(date +%Y%m%d-%H%M%S)}"
START_TIMEOUT_S="${START_TIMEOUT_S:-300}"

mkdir -p "$OUT_DIR"

gate_json="$OUT_DIR/probe_gate.json"
compose_project="lfm-w4-probe-$(date +%s)"
compose_for_run="$OUT_DIR/docker-compose.w4a16-gptq.yml"
cp "$COMPOSE_FILE" "$compose_for_run"

gate_set() {
  python3 - "$gate_json" "$1" "$2" <<'PY'
import json, sys
path, key, raw = sys.argv[1:]
try:
    data = json.load(open(path))
except FileNotFoundError:
    data = {"gates": {}, "notes": []}
value = True if raw == "true" else False if raw == "false" else raw
data.setdefault("gates", {})[key] = value
with open(path, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
PY
}

gate_note() {
  python3 - "$gate_json" "$1" <<'PY'
import json, sys
path, note = sys.argv[1:]
try:
    data = json.load(open(path))
except FileNotFoundError:
    data = {"gates": {}, "notes": []}
data.setdefault("notes", []).append(note)
with open(path, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
PY
}

gate_set artifact_valid false
gate_set compose_valid false
gate_set accuracy_gate_passed false
gate_set docker_build_passed false
gate_set cold_start_passed false
gate_set health_passed false
gate_set model_alias_passed false
gate_set trace_replay_passed false
gate_set zero_failures false
gate_set no_corrupted_output false

if ! python3 scripts/validate_w4_candidate.py --artifact "$ARTIFACT_DIR" \
  > "$OUT_DIR/w4_artifact_validation.json"; then
  cat "$OUT_DIR/w4_artifact_validation.json" >&2
  gate_note "W4 artifact validation failed. For the exact Docker gate, submission/model must contain the no-conv W4 checkpoint."
  exit 1
fi
gate_set artifact_valid true

if ! python3 scripts/validate_submission_compose.py "$COMPOSE_FILE" \
  --candidate w4a16-gptq > "$OUT_DIR/compose_validation.txt"; then
  cat "$OUT_DIR/compose_validation.txt" >&2
  gate_note "Compose validation failed. Replace the placeholder image tag before running the exact Docker gate."
  exit 1
fi
gate_set compose_valid true

if [[ "$RUN_ACCURACY" == "1" ]]; then
  gate_note "RUN_ACCURACY=1 requested, but this repository's automated W4 accuracy helper requires a live server config rather than the final Docker compose. Run ARC Challenge full, GSM8K limit 200, GPQA Diamond official, and write ACCURACY_GATE_JSON with pass=true before submitting."
  exit 1
fi

if [[ -z "$ACCURACY_GATE_JSON" ]]; then
  gate_note "Missing ACCURACY_GATE_JSON. Required: ARC Challenge full, GSM8K limit 200, GPQA Diamond official if time permits, plus output coherence/format."
  echo "Missing ACCURACY_GATE_JSON; refusing Docker probe gate." >&2
  exit 1
fi

python3 - "$ACCURACY_GATE_JSON" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
if data.get("pass") is not True:
    raise SystemExit("accuracy gate JSON does not have pass=true")
if data.get("accuracy_drop_risk") not in (False, 0, "false", "False", None):
    raise SystemExit("accuracy_drop_risk is set; do not submit W4")
PY
cp "$ACCURACY_GATE_JSON" "$OUT_DIR/accuracy_gate.json"
gate_set accuracy_gate_passed true

if ! command -v docker >/dev/null 2>&1; then
  gate_note "Docker CLI is not available on this host."
  echo "Docker CLI is not available; cannot run exact Docker gate." >&2
  exit 1
fi

docker build -t "$IMAGE_TAG" . 2>&1 | tee "$OUT_DIR/docker_build.log"
gate_set docker_build_passed true

python3 - "$compose_for_run" "$IMAGE_TAG" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
image = sys.argv[2]
text = path.read_text()
text = text.replace("<DOCKERHUB_USER>/lfm-serving:w4a16-gptq", image)
path.write_text(text)
PY

cleanup() {
  docker compose -p "$compose_project" -f "$compose_for_run" down -v \
    > "$OUT_DIR/docker_compose_down.log" 2>&1 || true
}
trap cleanup EXIT

docker compose -p "$compose_project" -f "$compose_for_run" up -d \
  > "$OUT_DIR/docker_compose_up.log" 2>&1

ready=0
deadline=$((SECONDS + START_TIMEOUT_S))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  docker compose -p "$compose_project" -f "$compose_for_run" logs --no-color model \
    > "$OUT_DIR/container.log" 2>&1 || true
  if curl -fsS "$BASE_URL/health" > "$OUT_DIR/health.txt"; then
    ready=1
    break
  fi
  if grep -qiE "Traceback|CUDA out of memory|MacheteLinearKernel requires|KeyError" "$OUT_DIR/container.log"; then
    break
  fi
  sleep 2
done

if [[ "$ready" != "1" ]]; then
  gate_note "Container did not become healthy. If startup failed on H200, likely a Linear shape is unsupported by Machete; do not spend a second W4 slot."
  echo "Container did not become healthy." >&2
  exit 1
fi
gate_set cold_start_passed true
gate_set health_passed true

curl -fsS "$BASE_URL/v1/models" > "$OUT_DIR/models.json"
python3 - "$OUT_DIR/models.json" "$MODEL_NAME" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
want = sys.argv[2]
ids = [m.get("id") for m in data.get("data", [])]
if want not in ids:
    raise SystemExit(f"model alias {want!r} not found in {ids!r}")
PY
gate_set model_alias_passed true

source .venv/bin/activate
python3 benchmark/replay_trace.py \
  --trace "$TRACE" \
  --workload "$WORKLOAD" \
  --model "$MODEL_NAME" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --base-url "$BASE_URL" \
  --name "$(basename "$OUT_DIR")" \
  --out-dir "$OUT_DIR/replay" \
  --keep-output-text \
  --verbose 2>&1 | tee "$OUT_DIR/replay.log"
gate_set trace_replay_passed true

python3 - "$OUT_DIR/replay/results.jsonl" <<'PY'
import json, sys
bad = []
corrupt = []
for line in open(sys.argv[1]):
    row = json.loads(line)
    text = row.get("output_text") or ""
    if row.get("status") != "ok":
        bad.append(row)
    if not text.strip() or "\ufffd" in text or "\x00" in text:
        corrupt.append(row)
if bad:
    raise SystemExit(f"{len(bad)} failed replay rows")
if corrupt:
    raise SystemExit(f"{len(corrupt)} corrupted/empty output rows")
PY
gate_set zero_failures true
gate_set no_corrupted_output true

python3 - "$gate_json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
data["pass"] = all(v is True for v in data.get("gates", {}).values())
with open(path, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
print(json.dumps(data, indent=2, sort_keys=True))
PY

echo "W4 probe gates passed: $gate_json"
