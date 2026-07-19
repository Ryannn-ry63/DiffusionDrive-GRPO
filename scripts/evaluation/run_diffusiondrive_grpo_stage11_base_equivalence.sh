#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [LIMIT=256] [CUDA_DEVICE=0]"
  exit 2
fi

LIMIT="${1:-256}"
CUDA_DEVICE="${2:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TOKENS_FILE="$ROOT_DIR/artifacts/grpo_stage0/holdout1024_aligned_t8.json"
OUTPUT_DIR="$ROOT_DIR/artifacts/grpo_stage11/base_equivalence"
CURRENT_ARTIFACT="$OUTPUT_DIR/base_current_${LIMIT}.json"
REFERENCE_ARTIFACT="$OUTPUT_DIR/base_reference_${LIMIT}.json"

mkdir -p "$OUTPUT_DIR"
"$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage11_eval.sh" \
  current "$BASE_CHECKPOINT" "$CURRENT_ARTIFACT" "$TOKENS_FILE" "$LIMIT" \
  none default default val "$CUDA_DEVICE"
"$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage11_eval.sh" \
  reference "$BASE_CHECKPOINT" "$REFERENCE_ARTIFACT" "$TOKENS_FILE" "$LIMIT" \
  none default default val "$CUDA_DEVICE"

/root/miniconda3/envs/navsim/bin/python \
  "$ROOT_DIR/scripts/evaluation/check_grpo_stage11_gate.py" \
  --gate base-equivalence \
  --baseline-artifacts "$CURRENT_ARTIFACT" \
  --candidate-artifacts "$REFERENCE_ARTIFACT" \
  --output "$OUTPUT_DIR/gate_${LIMIT}.json"
