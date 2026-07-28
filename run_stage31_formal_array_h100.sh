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
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
JOB_SCRIPT="$ROOT_DIR/run_stage31_cv_job_h100.sh"
TRAIN_SCRIPT="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage31_cv.sh"
PLAN="$ROOT_DIR/GRPO_STAGE31_SELECTOR_CONSISTENT_DECISION_GRPO_PLAN_20260725.md"
AUDIT="$ROOT_DIR/artifacts/grpo_stage31/cv/training/DPF/fold0/audit/audit.json"
TIMEOUT_SECONDS="${STAGE31_WAIT_TIMEOUT_SECONDS:-21600}"
PLAN_SHA256=3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d

[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || { echo "invalid timeout"; exit 2; }
[[ -d "$ROOT_DIR" ]] || { echo "missing Stage31 root: $ROOT_DIR"; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN"; exit 2; }
for path in "$JOB_SCRIPT" "$TRAIN_SCRIPT" "$PLAN"; do
  [[ -f "$path" ]] || { echo "missing required Stage31 path: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA256" ]] || {
  echo "Stage31 plan SHA256 mismatch: $PLAN"; exit 2;
}
bash -n "$JOB_SCRIPT" "$TRAIN_SCRIPT"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage31 formal training requires exactly 8 GPUs"; exit 2; }

deadline=$((SECONDS + TIMEOUT_SECONDS))
echo "Stage31 job $JOB_INDEX maps to branch=$BRANCH holdout=$HOLDOUT."
echo "Waiting for the passing Stage31 DPF/fold0 one-step audit..."
while true; do
  if [[ -f "$AUDIT" ]] && \
    "$PYTHON_BIN" - "$AUDIT" >/dev/null 2>&1 <<'PY'
import json
import sys

p = json.load(open(sys.argv[1]))
checks = (
    p.get("passed"), p.get("stage") == 31, p.get("phase") == "audit",
    p.get("branch") == "DPF", p.get("holdout_fold") == 0,
    p.get("num_logged_optimizer_steps") == 1,
    p.get("active_decoder_gradients"), p.get("zero_frozen_gradients"),
    p.get("frozen_reference_bitwise_equal_public"),
    p.get("frozen_training_selector_bitwise_equal"),
    p.get("exact_global_bucket_sampler"),
)
if not all(checks):
    raise RuntimeError("Stage31 audit is not a passing gate")
PY
  then
    break
  fi
  (( SECONDS < deadline )) || { echo "timed out waiting for Stage31 audit"; exit 1; }
  echo "Audit is not ready yet; retrying in 15 seconds..."
  sleep 15
done

echo "Audit passed; starting Stage31 formal job=$JOB_INDEX branch=$BRANCH holdout=$HOLDOUT."
exec bash "$JOB_SCRIPT" formal "$BRANCH" "$HOLDOUT"
