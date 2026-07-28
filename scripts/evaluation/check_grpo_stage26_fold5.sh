#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
FOLD5="$ROOT_DIR/artifacts/grpo_stage26/fold5"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold5_manifest.json"
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
FINAL=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage26_generator_final_folds0_4_seed0/2026.07.23.15.32.57/lightning_logs/version_0/checkpoints/grpo-01-160.ckpt

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/check_grpo_stage26_fold5.py" \
  --manifest "$MANIFEST" \
  --calibration "$CALIBRATION" \
  --final-checkpoint "$FINAL" \
  --b -1 "$FOLD5/B_default.json" \
  --b 20260823 "$FOLD5/B_ns20260823.json" \
  --b 20260824 "$FOLD5/B_ns20260824.json" \
  --b 20260825 "$FOLD5/B_ns20260825.json" \
  --c -1 "$FOLD5/C_default.json" \
  --c 20260823 "$FOLD5/C_ns20260823.json" \
  --c 20260824 "$FOLD5/C_ns20260824.json" \
  --c 20260825 "$FOLD5/C_ns20260825.json" \
  --output "$FOLD5/final_report.json"
