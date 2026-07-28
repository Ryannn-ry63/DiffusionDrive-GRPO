#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 JOB_INDEX(0..7)"
  echo "  0..3 => DP folds 0..3; 4..7 => DPF folds 0..3"
  exit 2
fi
JOB_INDEX="$1"
[[ "$JOB_INDEX" =~ ^[0-7]$ ]] || { echo "job index must be 0..7"; exit 2; }
if (( JOB_INDEX < 4 )); then
  BRANCH=DP
  HOLDOUT="$JOB_INDEX"
else
  BRANCH=DPF
  HOLDOUT="$((JOB_INDEX - 4))"
fi

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
JOB_SCRIPT="$ROOT_DIR/run_stage31_cv_eval_job_h100.sh"
CELL_SCRIPT="$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage31_cv_eval_cell.sh"
GATE_SCRIPT="$ROOT_DIR/run_stage31_cv_gate.sh"
AUDIT="$ROOT_DIR/artifacts/grpo_stage31/cv/training/$BRANCH/fold$HOLDOUT/formal/audit.json"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"

[[ -d "$ROOT_DIR" ]] || { echo "missing Stage31 root: $ROOT_DIR"; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN"; exit 2; }
for path in "$JOB_SCRIPT" "$CELL_SCRIPT" "$GATE_SCRIPT" "$AUDIT"; do
  [[ -f "$path" ]] || { echo "missing Stage31 evaluation path: $path"; exit 2; }
done
bash -n "$JOB_SCRIPT" "$CELL_SCRIPT" "$GATE_SCRIPT"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage31 evaluation requires exactly 8 GPUs"; exit 2; }

"$PYTHON_BIN" - "$AUDIT" "$BRANCH" "$HOLDOUT" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))
branch, fold = sys.argv[2], int(sys.argv[3])
if not (
    p.get("passed") and p.get("stage") == 31 and p.get("phase") == "formal"
    and p.get("branch") == branch and p.get("holdout_fold") == fold
    and [item["global_step"] for item in p.get("checkpoints", [])]
    == [48, 96, 144, 192]
):
    raise RuntimeError("Stage31 formal audit is not a passing evaluation input")
PY

echo "Starting Stage31 evaluation job=$JOB_INDEX branch=$BRANCH holdout=$HOLDOUT."
exec bash "$JOB_SCRIPT" "$BRANCH" "$HOLDOUT"
