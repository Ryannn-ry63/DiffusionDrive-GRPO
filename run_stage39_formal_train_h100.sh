#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 BC|STD|SET FOLD(0|1|3) [RUN_ID]"
  exit 2
fi

BRANCH="${1^^}"
FOLD="$2"
RUN_ID="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
TRAIN="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage39_challenger.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
DEVICES="${STAGE39_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
PLAN="$ROOT_DIR/GRPO_STAGE39_40_NON_DESTRUCTIVE_CHALLENGER_PLAN_20260729.md"
PLAN_SHA="7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"
SELECTION="${STAGE39_SELECTION:-$ROOT_DIR/artifacts/grpo_stage39/pilot/selection.json}"
PHASE0="$ROOT_DIR/artifacts/grpo_stage39/phase0/audit.json"

[[ "$BRANCH" =~ ^(BC|STD|SET)$ ]] || { echo "branch must be BC, STD, or SET"; exit 2; }
[[ "$FOLD" =~ ^(0|1|3)$ ]] || { echo "formal fold must be 0, 1, or 3"; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_ID"; exit 2; }
[[ "$DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage39 requires exactly eight GPU indices"; exit 2;
}
[[ -d "$ROOT_DIR" && -x "$TRAIN" && -x "$PYTHON_BIN" && -f "$PLAN" ]] || {
  echo "Stage39 formal paths are incomplete under $ROOT_DIR"; exit 2;
}
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA" ]] || {
  echo "Stage39 plan SHA drifted"; exit 2;
}

export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

wait_for_artifact() {
  local path="$1"
  local label="$2"
  local waited=0
  local timeout="${STAGE39_WAIT_SECONDS:-259200}"
  while (( waited < timeout )); do
    if [[ -f "$path" ]]; then
      echo "Found $label: $path"
      return 0
    fi
    if (( waited == 0 || waited % 300 == 0 )); then
      echo "Waiting for $label (${waited}s/${timeout}s): $path"
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "Timed out waiting for $label: $path"
  exit 2
}

validate_gate() {
  "$PYTHON_BIN" - "$PHASE0" "$SELECTION" "$PLAN_SHA" <<'PY'
import json
import sys
from pathlib import Path

phase0, selection, plan = (json.loads(Path(p).read_text()) for p in sys.argv[1:4])
assert phase0.get("stage") == 39 and phase0.get("passed") is True
assert phase0.get("plan_sha256") == plan
assert selection.get("stage") == 39
assert selection.get("phase") == "pilot_selection"
assert selection.get("passed") is True
assert selection.get("plan_sha256") == plan
assert int(selection.get("selected_step", -1)) in (48, 96, 192)
PY
}

stop_gpu_occupy() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t gpu-occupy 2>/dev/null; then
    echo "Stopping fixed gpu-occupy session before Stage39 formal training..."
    tmux kill-session -t gpu-occupy
    for _ in $(seq 1 20); do
      pgrep -f '[o]ccupy_multi.py' >/dev/null || break
      sleep 1
    done
    if pgrep -f '[o]ccupy_multi.py' >/dev/null; then
      echo "occupy_multi.py did not exit; refusing to start Stage39 training"
      exit 2
    fi
    echo "gpu-occupy stopped and memory release confirmed."
    nvidia-smi --query-gpu=index,memory.used,memory.total \
      --format=csv,noheader,nounits || true
  fi
}

cd "$ROOT_DIR"
FORMAL_EXP="stage39_${BRANCH,,}_formal_f${FOLD}_${RUN_ID}"
wait_for_artifact "$PHASE0" "Stage39 Phase0 audit"
wait_for_artifact "$SELECTION" "Stage39 pilot selection"
validate_gate
GRPO_STAGE39_PREFLIGHT_ONLY=1 "$TRAIN" formal "$BRANCH" "$FOLD" \
  "${FORMAL_EXP}_preflight" "$DEVICES" >/dev/null

PILOT_AUDIT="$ROOT_DIR/artifacts/grpo_stage39/audit/training/$BRANCH/fold2/audit.json"
wait_for_artifact "$PILOT_AUDIT" "Stage39 Phase1 audit ($BRANCH)"
"$PYTHON_BIN" - "$PILOT_AUDIT" "$BRANCH" "$PLAN_SHA" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p.get("stage") == 39 and p.get("phase") == "audit"
assert p.get("branch") == sys.argv[2] and p.get("holdout_fold") == 2
assert p.get("passed") is True and p.get("plan_sha256") == sys.argv[3]
PY

FORMAL_DIR="$ROOT_DIR/artifacts/grpo_stage39/formal/training/$BRANCH/fold${FOLD}"
if [[ -e "$FORMAL_DIR/checkpoints.json" || -e "$FORMAL_DIR/audit.json" ]]; then
  echo "Stage39 formal artifacts already exist; refusing to overwrite: $FORMAL_DIR"
  exit 2
fi
stop_gpu_occupy
"$TRAIN" formal "$BRANCH" "$FOLD" "$FORMAL_EXP" "$DEVICES"

echo "PASS Stage39 formal training branch=$BRANCH fold=$FOLD"
echo "Formal experiment: $EXP_ROOT/$FORMAL_EXP"
echo "Freeze: $FORMAL_DIR/checkpoints.json"
