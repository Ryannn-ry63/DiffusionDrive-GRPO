#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 COV|MCC HOLDOUT_FOLD(0..3)"
  exit 2
fi
BRANCH="$1"
HOLDOUT="$2"
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
AUDIT="$ROOT_DIR/artifacts/grpo_stage30/cv/training/MCC/fold0/audit/audit.json"
TIMEOUT_SECONDS="${STAGE30_WAIT_TIMEOUT_SECONDS:-21600}"

[[ "$BRANCH" == COV || "$BRANCH" == MCC ]] || { echo "bad branch"; exit 2; }
[[ "$HOLDOUT" =~ ^[0-3]$ ]] || { echo "holdout must be 0..3"; exit 2; }
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || { echo "invalid timeout"; exit 2; }
[[ -d "$ROOT_DIR" ]] || { echo "missing Stage30 root: $ROOT_DIR"; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN"; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 8 ]] || { echo "Stage30 formal training requires exactly 8 GPUs"; exit 2; }

deadline=$((SECONDS + TIMEOUT_SECONDS))
echo "Waiting for the passing Stage30 MCC/fold0 one-step audit..."
while true; do
  if [[ -f "$AUDIT" ]] && \
    "$PYTHON_BIN" - "$AUDIT" >/dev/null 2>&1 <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
checks = (
    p.get('passed'), p.get('stage') == 30, p.get('phase') == 'audit',
    p.get('branch') == 'MCC', p.get('holdout_fold') == 0,
    p.get('num_logged_optimizer_steps') == 1,
    p.get('active_decoder_gradients'), p.get('zero_frozen_gradients'),
    p.get('frozen_reference_bitwise_equal_public'),
    p.get('frozen_training_selector_bitwise_equal'),
    p.get('exact_global_bucket_sampler'),
)
if not all(checks):
    raise RuntimeError('Stage30 audit is not a passing gate')
PY
  then
    break
  fi
  (( SECONDS < deadline )) || { echo "timed out waiting for Stage30 audit"; exit 1; }
  echo "Audit is not ready yet; retrying in 15 seconds..."
  sleep 15
done

echo "Audit passed; starting Stage30 formal branch=$BRANCH holdout=$HOLDOUT."
exec bash "$ROOT_DIR/run_stage30_cv_job_h100.sh" formal "$BRANCH" "$HOLDOUT"
