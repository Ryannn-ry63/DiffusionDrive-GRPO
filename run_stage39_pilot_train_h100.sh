#!/usr/bin/env bash
set -euo pipefail

# Queue-safe Stage39 pilot launcher.  It is intentionally a thin, absolute-path
# wrapper around the canonical training script: jobs submitted before Phase0
# finishes wait for the immutable Phase0 gate instead of failing early.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 BC|STD|SET [RUN_ID]"
  exit 2
fi

BRANCH="${1^^}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
TRAIN="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage39_challenger.sh"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
DEVICES="${STAGE39_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
PHASE0="$ROOT_DIR/artifacts/grpo_stage39/phase0/audit.json"
PLAN="$ROOT_DIR/GRPO_STAGE39_40_NON_DESTRUCTIVE_CHALLENGER_PLAN_20260729.md"
PLAN_SHA="7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"

[[ "$BRANCH" =~ ^(BC|STD|SET)$ ]] || { echo "branch must be BC, STD, or SET"; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_ID"; exit 2; }
[[ "$DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage39 requires exactly eight GPU indices"; exit 2;
}
[[ -d "$ROOT_DIR" && -x "$TRAIN" && -x "$PYTHON_BIN" && -f "$PLAN" ]] || {
  echo "Stage39 pilot paths are incomplete under $ROOT_DIR"; exit 2;
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

wait_for_phase0() {
  local waited=0
  local timeout="${STAGE39_WAIT_SECONDS:-259200}"
  while (( waited < timeout )); do
    if [[ -f "$PHASE0" ]] && "$PYTHON_BIN" - "$PHASE0" "$PLAN_SHA" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if (
    payload.get("stage") == 39
    and payload.get("phase") == "offline_target_audit"
    and payload.get("passed") is True
    and payload.get("plan_sha256") == sys.argv[2]
) else 1)
PY
    then
      echo "Stage39 Phase0 gate passed: $PHASE0"
      return 0
    fi
    if (( waited == 0 || waited % 300 == 0 )); then
      echo "Waiting for a passing Stage39 Phase0 gate (${waited}s/${timeout}s)..."
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "Timed out waiting for Stage39 Phase0: $PHASE0"
  exit 2
}

# Only the fixed occupancy session is touched.  This is required immediately
# before the first real GPU process, after all queue-safe waiting/preflight.
stop_gpu_occupy() {
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t gpu-occupy 2>/dev/null; then
    echo "Stopping fixed gpu-occupy session before Stage39 pilot training..."
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
AUDIT_EXP="stage39_${BRANCH,,}_pilot_audit_${RUN_ID}"
PILOT_EXP="stage39_${BRANCH,,}_pilot_formal_${RUN_ID}"

# Resolve Hydra and all paths before waiting or consuming a GPU.
GRPO_STAGE39_PREFLIGHT_ONLY=1 "$TRAIN" audit "$BRANCH" 2 \
  "${AUDIT_EXP}_preflight" "$DEVICES" >/dev/null
GRPO_STAGE39_PREFLIGHT_ONLY=1 "$TRAIN" pilot "$BRANCH" 2 \
  "${PILOT_EXP}_preflight" "$DEVICES" >/dev/null
wait_for_phase0

AUDIT_DIR="$ROOT_DIR/artifacts/grpo_stage39/audit/training/$BRANCH/fold2"
AUDIT_JSON="$AUDIT_DIR/audit.json"
AUDIT_FREEZE="$AUDIT_DIR/checkpoints.json"
reuse_audit=0
if [[ -e "$AUDIT_JSON" || -e "$AUDIT_FREEZE" ]]; then
  if [[ ! -f "$AUDIT_JSON" || ! -f "$AUDIT_FREEZE" ]]; then
    echo "Partial Stage39 pilot audit exists; refusing overwrite: $AUDIT_DIR"
    exit 2
  fi
  if "$PYTHON_BIN" - "$AUDIT_JSON" "$BRANCH" "$PLAN_SHA" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if (
    p.get("stage") == 39 and p.get("phase") == "audit"
    and p.get("branch") == sys.argv[2] and p.get("holdout_fold") == 2
    and p.get("passed") is True and p.get("plan_sha256") == sys.argv[3]
) else 1)
PY
  then
    reuse_audit=1
    echo "Reusing passed Stage39 pilot audit: $AUDIT_JSON"
  else
    echo "Existing Stage39 pilot audit is invalid; refusing to overwrite: $AUDIT_JSON"
    exit 2
  fi
fi
if [[ "$reuse_audit" == 0 ]]; then
  stop_gpu_occupy
  "$TRAIN" audit "$BRANCH" 2 "$AUDIT_EXP" "$DEVICES"
fi

PILOT_DIR="$ROOT_DIR/artifacts/grpo_stage39/pilot/training/$BRANCH/fold2"
if [[ -e "$PILOT_DIR/checkpoints.json" || -e "$PILOT_DIR/audit.json" ]]; then
  echo "Stage39 pilot artifacts already exist; refusing to overwrite: $PILOT_DIR"
  exit 2
fi
stop_gpu_occupy
"$TRAIN" pilot "$BRANCH" 2 "$PILOT_EXP" "$DEVICES"

echo "PASS Stage39 pilot training branch=$BRANCH"
echo "Audit experiment: $EXP_ROOT/$AUDIT_EXP"
echo "Pilot experiment: $EXP_ROOT/$PILOT_EXP"
echo "Freeze: $PILOT_DIR/checkpoints.json"
