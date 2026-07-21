#!/bin/bash
# Prepare submission/model from a validated W4 candidate or staged H200 backend
# probe.
#
# Default mode refuses to proceed unless candidate_report JSON has
# gates.candidate_passes true. Probe mode is intentionally separate: it accepts
# an accuracy-gate JSON and stages the W4 checkpoint for the exact Docker gate.
# The Docker gate still must be run after staging before any official submit.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="candidate"
if [[ "${1:-}" == "--probe" ]]; then
  MODE="probe"
  shift
fi

REPORT_JSON="${1:?usage: prepare_w4_submission.sh [--probe] <candidate_report.json|accuracy_gate.json> <w4_artifact_dir> [compose.yml]}"
ARTIFACT_DIR="${2:?usage: prepare_w4_submission.sh [--probe] <candidate_report.json|accuracy_gate.json> <w4_artifact_dir> [compose.yml]}"
COMPOSE_FILE="${3:-submission/docker-compose.w4a16-gptq.yml}"
TARGET_DIR="$PROJECT_DIR/submission/model"

python3 - <<'PY' "$REPORT_JSON" "$MODE"
import json, sys
path = sys.argv[1]
mode = sys.argv[2]
data = json.load(open(path))
if mode == "candidate":
    gates = data.get("gates") or {}
    if gates.get("candidate_passes") is not True:
        print(f"candidate report does not pass: {path}", file=sys.stderr)
        for key, value in gates.items():
            if value is not True:
                print(f"  {key}: {value}", file=sys.stderr)
        raise SystemExit(1)
else:
    if data.get("pass") is not True:
        print(f"accuracy gate does not pass: {path}", file=sys.stderr)
        raise SystemExit(1)
    if data.get("accuracy_drop_risk") not in (False, 0, "false", "False", None):
        print(f"accuracy_drop_risk is set in {path}; do not stage W4", file=sys.stderr)
        raise SystemExit(1)
PY

python3 scripts/validate_w4_candidate.py --artifact "$ARTIFACT_DIR" >/tmp/lfm_w4_artifact_validation.json
python3 scripts/validate_submission_compose.py "$COMPOSE_FILE" \
  --candidate w4a16-gptq --allow-placeholder-image >/tmp/lfm_w4_compose_validation.txt

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "artifact directory not found: $ARTIFACT_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
find "$TARGET_DIR" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
cp -a "$ARTIFACT_DIR"/. "$TARGET_DIR"/
touch "$TARGET_DIR/.gitkeep"

python3 scripts/validate_w4_candidate.py --artifact "$TARGET_DIR" \
  > "$PROJECT_DIR/submission/w4_model_validation.json"
python3 scripts/validate_submission_compose.py "$COMPOSE_FILE" \
  --candidate w4a16-gptq --allow-placeholder-image \
  > "$PROJECT_DIR/submission/w4_compose_validation.txt"

echo "Prepared W4 submission model in submission/model"
echo "Validation: submission/w4_model_validation.json"
echo "Compose validation: submission/w4_compose_validation.txt"
if [[ "$MODE" == "probe" ]]; then
  echo "Staged as an H200 backend probe, not as a latency-validated final candidate."
  echo "Do not submit until scripts/run_w4_probe_gates.sh passes against the exact Docker image."
fi
echo "Before final submission, replace the placeholder image in $COMPOSE_FILE and build/push Dockerfile."
