#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[01]$ ]]; then
  echo "Usage: $0 FOLD(0|1) [RUN_ID]"
  exit 2
fi

FOLD="$1"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
TRAIN="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage38_escr.sh"
[[ -d "$ROOT_DIR" && -x "$TRAIN" ]] || {
  echo "Stage38 repository/training script is missing under $ROOT_DIR"
  exit 2
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_ID"; exit 2; }

if [[ "$FOLD" == 1 ]]; then
  KILL_SELECTION="$ROOT_DIR/artifacts/grpo_stage38/pilot/kill_selection.json"
  [[ -f "$KILL_SELECTION" ]] || {
    echo "Fold1 is locked until the fold0 Stage38 kill-test passes"
    exit 2
  }
fi

# Obey the repository GPU-occupancy contract. Only the fixed gpu-occupy
# session is touched; unrelated tmux sessions and processes are never killed.
if command -v tmux >/dev/null 2>&1 \
    && tmux has-session -t gpu-occupy 2>/dev/null; then
  echo "Stopping fixed gpu-occupy session before Stage38 training..."
  tmux kill-session -t gpu-occupy
  for _ in $(seq 1 20); do
    if ! pgrep -f '[o]ccupy_multi.py' >/dev/null; then break; fi
    sleep 1
  done
  if pgrep -f '[o]ccupy_multi.py' >/dev/null; then
    echo "occupy_multi.py did not exit; refusing to start GPU training"
    exit 2
  fi
  echo "gpu-occupy stopped and occupy_multi.py exited."
  nvidia-smi --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader,nounits || true
fi

export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp"
export OPENSCENE_DATA_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/global_public/public_datas/NAVSIM/maps"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"

AUDIT_EXP="stage38_escr_f${FOLD}_audit_${RUN_ID}"
FORMAL_EXP="stage38_escr_f${FOLD}_formal_${RUN_ID}"
cd "$ROOT_DIR"

GRPO_STAGE38_PREFLIGHT_ONLY=1 \
  "$TRAIN" audit "$FOLD" "${AUDIT_EXP}_preflight" 0,1,2,3,4,5,6,7
GRPO_STAGE38_PREFLIGHT_ONLY=1 \
  "$TRAIN" formal "$FOLD" "${FORMAL_EXP}_preflight" 0,1,2,3,4,5,6,7

AUDIT_ARTIFACT="$ROOT_DIR/artifacts/grpo_stage38/pilot/training/ESCR/fold${FOLD}/audit/audit.json"
AUDIT_FREEZE="$ROOT_DIR/artifacts/grpo_stage38/pilot/training/ESCR/fold${FOLD}/audit/checkpoints.json"
reuse_audit=0
if [[ -f "$AUDIT_ARTIFACT" && -f "$AUDIT_FREEZE" ]]; then
  if "$PYTHON_BIN" - "$AUDIT_ARTIFACT" "$FOLD" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if not (
    payload.get("stage") == 38
    and payload.get("phase") == "audit"
    and payload.get("holdout_fold") == int(sys.argv[2])
    and payload.get("passed") is True
):
    raise SystemExit(1)
PY
  then
    reuse_audit=1
    echo "Reusing passed Stage38 audit artifact: $AUDIT_ARTIFACT"
  fi
fi
if [[ "$reuse_audit" == 0 ]]; then
  "$TRAIN" audit "$FOLD" "$AUDIT_EXP" 0,1,2,3,4,5,6,7
fi
"$TRAIN" formal "$FOLD" "$FORMAL_EXP" 0,1,2,3,4,5,6,7

echo "PASS Stage38 ESCR audit+formal training fold=$FOLD"
echo "Audit experiment: $NAVSIM_EXP_ROOT/$AUDIT_EXP"
echo "Formal experiment: $NAVSIM_EXP_ROOT/$FORMAL_EXP"
echo "Freeze: $ROOT_DIR/artifacts/grpo_stage38/pilot/training/ESCR/fold${FOLD}/formal/checkpoints.json"
