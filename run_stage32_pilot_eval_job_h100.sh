#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DPEL|SCF HOLDOUT(0..3)"
  exit 2
fi
BRANCH="$1"; HOLDOUT="$2"
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export NAVSIM_DEVKIT_ROOT="$ROOT_DIR/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
[[ "$BRANCH" == DPEL || "$BRANCH" == SCF ]] || { echo "bad branch"; exit 2; }
[[ "$HOLDOUT" =~ ^[0-3]$ ]] || { echo "bad holdout"; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage32 evaluation requires 8 GPUs"; exit 2; }
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_DIR/artifacts/grpo_stage32/h100_logs/${RUN_TAG}_eval_${BRANCH}_fold${HOLDOUT}"
mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"
AUDIT="$ROOT_DIR/artifacts/grpo_stage32/pilot/training/$BRANCH/fold$HOLDOUT/formal/audit.json"
"$PYTHON_BIN" - "$AUDIT" "$BRANCH" "$HOLDOUT" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1]))
if not (
    p.get("passed") and p.get("stage") == 32 and p.get("phase") == "formal"
    and p.get("branch") == sys.argv[2]
    and p.get("holdout_fold") == int(sys.argv[3])
    and [item["global_step"] for item in p.get("checkpoints", [])]
    == [192, 384, 576]
):
    raise RuntimeError("Stage32 formal audit is not a passing evaluation input")
PY
failed=0
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  bash scripts/evaluation/run_diffusiondrive_grpo_stage32_pilot_eval_cell.sh \
    "$BRANCH" "$HOLDOUT" "$gpu" "$gpu" >"$LOG_DIR/job${gpu}.log" 2>&1 &
  pids+=("$!")
done
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "PASS job${index}"
  else
    status=$?
    echo "FAIL job${index} exit=$status"
    failed=1
  fi
done
(( failed == 0 )) || { echo "inspect $LOG_DIR"; exit 1; }
echo "PASS Stage32 pilot evaluation branch=$BRANCH fold=$HOLDOUT"
echo "Logs: $LOG_DIR"
