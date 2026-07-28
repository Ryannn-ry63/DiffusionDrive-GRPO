#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 FINAL_STAGE25_SELECTOR_CHECKPOINT"
  exit 2
fi
SELECTOR="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
COLLECT_DIR="$ROOT_DIR/artifacts/grpo_stage26/fold4/collect"
OUTPUT="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
DOMAINS=(official_base stage16_epoch8 stage19_epoch8 stage21_epoch8 stage25_epoch2)
NOISES=(20260821 20260822)
ARGS=()
for domain in "${DOMAINS[@]}"; do
  for noise in "${NOISES[@]}"; do
    path="$COLLECT_DIR/${domain}_ns${noise}.json"
    [[ -f "$path" ]] || { echo "missing Stage26 fold4 collection: $path"; exit 2; }
    ARGS+=(--artifact "$path")
  done
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/calibrate_grpo_stage25_selector.py" \
  "${ARGS[@]}" --selector-checkpoint "$SELECTOR" \
  --bootstrap-seed 20260926 --output "$OUTPUT"
"$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1]))
assert p["stage"] == 25 and p["passed"]
assert p["num_artifacts"] == 10 and p["num_scenes"] == 10210
assert len(p["selected"]["group_means"]) == 10
p["closure_stage"] = 26
open(sys.argv[1],"w").write(json.dumps(p,indent=2)+"\n")
' "$OUTPUT"
