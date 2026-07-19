#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 CALIBRATION_PATH [UPDATES=128] [SEED=0] [CUDA_DEVICE=0]"
  exit 2
fi

CALIBRATION_PATH="$1"
UPDATES="${2:-128}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$CALIBRATION_PATH" ]]; then
  echo "Calibration artifact does not exist: $CALIBRATION_PATH"
  exit 2
fi
if [[ "$UPDATES" != "8" && "$UPDATES" != "64" && "$UPDATES" != "128" ]]; then
  echo "Phase 5 permits only U8, diagnostic U64, or U128"
  exit 2
fi

export GENERATION_TRUST_PROJECTION_MODE=reference_mean_ball
export GENERATION_TRUST_CALIBRATION_PATH="$CALIBRATION_PATH"

exec bash "$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_ablation.sh" \
  D "$SEED" 0.0 4 "$CUDA_DEVICE" 0.1 0.001 0.1 uniform 2 \
  hierarchical none "$UPDATES" 2
