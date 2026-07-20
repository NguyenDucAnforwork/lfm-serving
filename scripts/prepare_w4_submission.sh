#!/bin/bash
# Prepare submission/model from a validated W4 candidate.
#
# Refuses to proceed unless the candidate report JSON has gates.candidate_passes
# true and the W4 artifact validates. This script replaces submission/model/*
# with the validated artifact contents, preserving submission/model/.gitkeep.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

REPORT_JSON="${1:?usage: prepare_w4_submission.sh <candidate_report.json> <w4_artifact_dir> [compose.yml]}"
ARTIFACT_DIR="${2:?usage: prepare_w4_submission.sh <candidate_report.json> <w4_artifact_dir> [compose.yml]}"
COMPOSE_FILE="${3:-submission/docker-compose.w4a16-gptq.yml}"
TARGET_DIR="$PROJECT_DIR/submission/model"

python3 - <<'PY' "$REPORT_JSON"
import json, sys
path = sys.argv[1]
data = json.load(open(path))
gates = data.get("gates") or {}
if gates.get("candidate_passes") is not True:
    print(f"candidate report does not pass: {path}", file=sys.stderr)
    for key, value in gates.items():
        if value is not True:
            print(f"  {key}: {value}", file=sys.stderr)
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
echo "Before final submission, replace the placeholder image in $COMPOSE_FILE and build/push Dockerfile."
