#!/usr/bin/env bash
set -euo pipefail

# Path-stable launcher for the external 8-GPU H100/4090 runner.
# Usage:
#   bash ./run_stage32_pilot_h100.sh formal SCF 0 stage32_pilot_SCF_fold0_formal_seed203200
# Optional fifth argument overrides the visible GPU list.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PILOT="$SCRIPT_DIR/scripts/training/run_diffusiondrive_grpo_stage32_pilot.sh"
[[ -f "$PILOT" ]] || { echo "missing Stage32 pilot: $PILOT" >&2; exit 2; }

PHASE="${1:-}"
BRANCH="${2:-}"
HOLDOUT="${3:-}"
EXPERIMENT="${4:-}"
CUDA_DEVICES="${5:-0,1,2,3,4,5,6,7}"

if [[ -z "$PHASE" || -z "$BRANCH" || -z "$HOLDOUT" || -z "$EXPERIMENT" ]]; then
  echo "Usage: $0 {audit|formal} {DPEL|SCF} {0|1} EXPERIMENT [CUDA_DEVICES]" >&2
  exit 2
fi

exec bash "$PILOT" "$PHASE" "$BRANCH" "$HOLDOUT" "$EXPERIMENT" "$CUDA_DEVICES"
