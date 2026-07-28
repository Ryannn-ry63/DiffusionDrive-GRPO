#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
DEPLOY_DIR="$ROOT_DIR/artifacts/grpo_stage26/fold4/deploy"
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold4_manifest.json"
CHECKPOINT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage25_generator_development8_seed0/2026.07.23.08.31.19/lightning_logs/version_0/checkpoints/grpo-01-128.ckpt

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/check_grpo_stage26_generator_gate.py" \
  --manifest "$MANIFEST" --calibration "$CALIBRATION" \
  --base 20260821 "$DEPLOY_DIR/official_base_ns20260821.json" \
  --base 20260822 "$DEPLOY_DIR/official_base_ns20260822.json" \
  --candidate 20260821 "$DEPLOY_DIR/stage25_epoch2_ns20260821.json" \
  --candidate 20260822 "$DEPLOY_DIR/stage25_epoch2_ns20260822.json" \
  --candidate-checkpoint "$CHECKPOINT" \
  --output "$ROOT_DIR/artifacts/grpo_stage26/fold4/generator_gate.json"
