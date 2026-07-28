#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 JOB_ID GPU"
  echo "JOB_ID 0-8: Stage16/19/21 fold3 suffixes; 9-11: Stage25 epoch2 folds0-3"
  exit 2
fi
JOB_ID="$1"
GPU="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[[ "$JOB_ID" =~ ^([0-9]|1[01])$ ]] || { echo "invalid JOB_ID"; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid GPU"; exit 2; }

OLD_DOMAINS=(stage16_epoch8 stage19_epoch8 stage21_epoch8)
OLD_CHECKPOINTS=(
  /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage16_full_chain_seed0_dev10/2026.07.21.06.24.01/lightning_logs/version_0/checkpoints/grpo-07-536.ckpt
  /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage19_selected_anchor_seed0_development10/2026.07.21.14.17.30/lightning_logs/version_0/checkpoints/grpo-07-536.ckpt
  /inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage21_development8_seed0/2026.07.22.03.16.23/lightning_logs/version_0/checkpoints/grpo-07-512.ckpt
)
NAMESPACES=(-1 20260811 20260812)
LABELS=(default ns20260811 ns20260812)

if (( JOB_ID < 9 )); then
  DOMAIN_INDEX=$((JOB_ID / 3))
  NS_INDEX=$((JOB_ID % 3))
  DOMAIN="${OLD_DOMAINS[$DOMAIN_INDEX]}"
  GENERATOR="${OLD_CHECKPOINTS[$DOMAIN_INDEX]}"
  NAMESPACE="${NAMESPACES[$NS_INDEX]}"
  LABEL="${LABELS[$NS_INDEX]}"
  MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold3_manifest.json"
  LIMIT=1019
  OUTPUT="$ROOT_DIR/artifacts/grpo_stage26/candidate_bank_fold3/${DOMAIN}_${LABEL}.json"
else
  NS_INDEX=$((JOB_ID - 9))
  DOMAIN=stage25_epoch2
  GENERATOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage25_generator_development8_seed0/2026.07.23.08.31.19/lightning_logs/version_0/checkpoints/grpo-01-128.ckpt
  NAMESPACE="${NAMESPACES[$NS_INDEX]}"
  LABEL="${LABELS[$NS_INDEX]}"
  MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
  LIMIT=4075
  OUTPUT="$ROOT_DIR/artifacts/grpo_stage26/candidate_bank/${DOMAIN}_${LABEL}.json"
fi

"$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage24_candidate_bank.sh" \
  "$MANIFEST" "$LIMIT" "$DOMAIN" "$GENERATOR" "$NAMESPACE" "$OUTPUT" "$GPU"
