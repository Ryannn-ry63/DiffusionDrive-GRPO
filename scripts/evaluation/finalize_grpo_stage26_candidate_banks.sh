#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
OLD_DIR="$ROOT_DIR/artifacts/grpo_stage24/candidate_bank"
SUFFIX_DIR="$ROOT_DIR/artifacts/grpo_stage26/candidate_bank_fold3"
OUTPUT_DIR="$ROOT_DIR/artifacts/grpo_stage26/candidate_bank"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/prepare_grpo_stage26_base_banks.py" \
  --source-dir "$ROOT_DIR/artifacts/grpo_stage23/candidate_bank" \
  --output-dir "$OUTPUT_DIR"

DOMAINS=(stage16_epoch8 stage19_epoch8 stage21_epoch8)
SHAS=(
  3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23
  f0bc08a76d5c3aa27f2f19fd46dcf569406bdd41ff5d9af0078fa75e12986c4b
  7a8d850280aa043ef41734fc9854e5fe8f4c2bb6376cf2d7274c1511d06a9492
)
NAMESPACES=(-1 20260811 20260812)
LABELS=(default ns20260811 ns20260812)
for domain_index in 0 1 2; do
  for ns_index in 0 1 2; do
    domain="${DOMAINS[$domain_index]}"
    label="${LABELS[$ns_index]}"
    "$PYTHON_BIN" \
      "$ROOT_DIR/scripts/evaluation/merge_grpo_stage26_candidate_bank.py" \
      --manifest "$MANIFEST" \
      --prefix "$OLD_DIR/${domain}_${label}.json" \
      --suffix "$SUFFIX_DIR/${domain}_${label}.json" \
      --domain "$domain" --namespace "${NAMESPACES[$ns_index]}" \
      --checkpoint-sha256 "${SHAS[$domain_index]}" \
      --output "$OUTPUT_DIR/${domain}_${label}.json"
  done
done

AUDIT_ARGS=()
for domain in official_base stage16_epoch8 stage19_epoch8 stage21_epoch8 stage25_epoch2; do
  for label in default ns20260811 ns20260812; do
    AUDIT_ARGS+=(--artifact "$OUTPUT_DIR/${domain}_${label}.json")
  done
done
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/audit_grpo_stage26_candidate_banks.py" \
  --manifest "$MANIFEST" "${AUDIT_ARGS[@]}" \
  --output "$ROOT_DIR/artifacts/grpo_stage26/candidate_bank_audit.json"
