#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
FREEZE="$ROOT_DIR/artifacts/grpo_stage37/jfi/training/formal/checkpoints.json"; BASE="$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json"
JFI=$("$PYTHON_BIN" -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p["checkpoints"][0]["path"])' "$FREEZE")
for h in 0 1; do
 DIR="$ROOT_DIR/artifacts/grpo_stage37/jfi/calibration/h${h}/collect"
 "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/calibrate_grpo_stage37_jfi.py" --holdout "$h" --selector-checkpoint "$JFI" --base-selector-calibration "$BASE" \
  --artifact "$DIR/P_ns20261511.json" --artifact "$DIR/P_ns20261512.json" --artifact "$DIR/BPD_ns20261511.json" --artifact "$DIR/BPD_ns20261512.json" \
  --output "$ROOT_DIR/artifacts/grpo_stage37/jfi/calibration/h${h}/calibration.json"
done
echo "PASS both Stage37 JFI cross-fitted calibrations"
