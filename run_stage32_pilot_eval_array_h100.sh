#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 JOB_INDEX(0..3)"
  echo "  0 DPEL fold0; 1 DPEL fold1; 2 SCF fold0; 3 SCF fold1"
  exit 2
fi
JOB_INDEX="$1"
[[ "$JOB_INDEX" =~ ^[0-3]$ ]] || { echo "job index must be 0..3"; exit 2; }
if (( JOB_INDEX < 2 )); then
  BRANCH=DPEL
  HOLDOUT="$JOB_INDEX"
else
  BRANCH=SCF
  HOLDOUT="$((JOB_INDEX - 2))"
fi
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
[[ -d "$ROOT_DIR" ]] || { echo "missing Stage32 root: $ROOT_DIR"; exit 2; }
exec bash "$ROOT_DIR/run_stage32_pilot_eval_job_h100.sh" "$BRANCH" "$HOLDOUT"
