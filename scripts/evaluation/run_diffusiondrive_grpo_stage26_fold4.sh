#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 collect|deploy SELECTOR CALIBRATION|- JOB_ID GPU"
  echo "collect JOB_ID 0-9; deploy JOB_ID 0-3"
  exit 2
fi
MODE="$1"; SELECTOR="$2"; CALIBRATION="$3"; JOB_ID="$4"; GPU="$5"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold4_manifest.json"
BASE=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model
STAGE16=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage16_full_chain_seed0_dev10/2026.07.21.06.24.01/lightning_logs/version_0/checkpoints/grpo-07-536.ckpt
STAGE19=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage19_selected_anchor_seed0_development10/2026.07.21.14.17.30/lightning_logs/version_0/checkpoints/grpo-07-536.ckpt
STAGE21=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage21_development8_seed0/2026.07.22.03.16.23/lightning_logs/version_0/checkpoints/grpo-07-512.ckpt
STAGE25=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage25_generator_development8_seed0/2026.07.23.08.31.19/lightning_logs/version_0/checkpoints/grpo-01-128.ckpt
NOISES=(20260821 20260822)

[[ "$MODE" == collect || "$MODE" == deploy ]] || { echo "invalid mode"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid GPU"; exit 2; }
if [[ "$MODE" == collect ]]; then
  [[ "$JOB_ID" =~ ^[0-9]$ ]] || { echo "collect JOB_ID must be 0-9"; exit 2; }
  DOMAINS=(official_base stage16_epoch8 stage19_epoch8 stage21_epoch8 stage25_epoch2)
  CHECKPOINTS=("$BASE" "$STAGE16" "$STAGE19" "$STAGE21" "$STAGE25")
  DOMAIN_INDEX=$((JOB_ID / 2))
  NS_INDEX=$((JOB_ID % 2))
  OUT_DIR="$ROOT_DIR/artifacts/grpo_stage26/fold4/collect"
else
  [[ "$JOB_ID" =~ ^[0-3]$ ]] || { echo "deploy JOB_ID must be 0-3"; exit 2; }
  DOMAINS=(official_base stage25_epoch2)
  CHECKPOINTS=("$BASE" "$STAGE25")
  DOMAIN_INDEX=$((JOB_ID / 2))
  NS_INDEX=$((JOB_ID % 2))
  OUT_DIR="$ROOT_DIR/artifacts/grpo_stage26/fold4/deploy"
fi
DOMAIN="${DOMAINS[$DOMAIN_INDEX]}"
GENERATOR="${CHECKPOINTS[$DOMAIN_INDEX]}"
NOISE="${NOISES[$NS_INDEX]}"
OUTPUT="$OUT_DIR/${DOMAIN}_ns${NOISE}.json"

"$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage25_selector_eval.sh" \
  "$MODE" "$GENERATOR" "$DOMAIN" "$SELECTOR" "$CALIBRATION" \
  "$MANIFEST" 1021 "$NOISE" "$OUTPUT" "$GPU"
