#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(kill|formal)$ ]]; then
  echo "Usage: $0 kill|formal"
  exit 2
fi

PHASE="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
if [[ "$PHASE" == kill ]]; then
  OUTPUT="$ROOT_DIR/artifacts/grpo_stage38/pilot/kill_summary.json"
  SELECTION="$ROOT_DIR/artifacts/grpo_stage38/pilot/kill_selection.json"
else
  OUTPUT="$ROOT_DIR/artifacts/grpo_stage38/pilot/summary.json"
  SELECTION="$ROOT_DIR/artifacts/grpo_stage38/pilot/selection.json"
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/summarize_grpo_stage38_escr.py" \
  --phase "$PHASE" \
  --stage38-root "$ROOT_DIR/artifacts/grpo_stage38" \
  --bucket-manifest \
    "$ROOT_DIR/artifacts/grpo_stage30/manifests/buckets_6119.json" \
  --output "$OUTPUT" \
  --selection-output "$SELECTION"

[[ -f "$SELECTION" ]] || {
  echo "Stage38 $PHASE gate did not pass; inspect $OUTPUT"
  exit 1
}
echo "PASS Stage38 $PHASE gate; selection=$SELECTION"
