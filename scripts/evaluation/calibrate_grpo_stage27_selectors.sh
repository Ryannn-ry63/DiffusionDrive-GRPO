#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
PLAN="$ROOT_DIR/GRPO_STAGE27_STAGE26_ALIGNED_EXECUTION_PLAN_20260724.md"
failed=0

for branch in public multi; do
  FREEZE="$ROOT_DIR/artifacts/grpo_stage27/selectors/$branch/formal/frozen_selector.json"
  CALIBRATION="$ROOT_DIR/artifacts/grpo_stage27/selectors/$branch/calibration.json"
  [[ -f "$FREEZE" ]] || { echo "missing formal selector freeze: $FREEZE"; exit 2; }
  [[ ! -e "$CALIBRATION" ]] || { echo "refusing to overwrite: $CALIBRATION"; exit 2; }
  SELECTOR="$("$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); assert p["passed"]; print(p["stage25_checkpoint"])
' "$FREEZE")"
  status=0
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/calibrate_grpo_stage25_selector.py" \
    --profile stage27_phase2 \
    --artifact "$ROOT_DIR/artifacts/grpo_stage27/selectors/$branch/fold4/collect_ns20260821.json" \
    --artifact "$ROOT_DIR/artifacts/grpo_stage27/selectors/$branch/fold4/collect_ns20260822.json" \
    --selector-checkpoint "$SELECTOR" --bootstrap-seed 20260927 \
    --output "$CALIBRATION" || status=$?
  if (( status != 0 && status != 2 )); then
    echo "Stage27 calibration crashed for branch=$branch status=$status"
    exit "$status"
  fi
  if (( status == 2 )); then
    echo "Stage27 calibration gate failed for branch=$branch"
    failed=1
  fi
done

SELECTION="$ROOT_DIR/artifacts/grpo_stage27/selectors/selection.json"
[[ ! -e "$SELECTION" ]] || { echo "refusing to overwrite: $SELECTION"; exit 2; }
status=0
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/select_grpo_stage27_selector_branch.py" \
  --public-calibration "$ROOT_DIR/artifacts/grpo_stage27/selectors/public/calibration.json" \
  --multi-calibration "$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json" \
  --public-freeze "$ROOT_DIR/artifacts/grpo_stage27/selectors/public/formal/frozen_selector.json" \
  --multi-freeze "$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/formal/frozen_selector.json" \
  --plan "$PLAN" --output "$SELECTION" || status=$?
if (( status != 0 )); then
  exit "$status"
fi
if (( failed != 0 )); then
  echo "At least one Stage27 branch failed, but the other passed selection."
fi
echo "Stage27 selected selector: $SELECTION"
