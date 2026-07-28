#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DEVELOPMENT_CHECKPOINT_DIR OUTPUT_DIR"
  exit 2
fi
CHECKPOINT_DIR="$1"; OUTPUT_DIR="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/calibration_manifest.json"
LIMIT=1021

[[ -d "$CHECKPOINT_DIR" && -f "$BASE_CHECKPOINT" && -f "$MANIFEST" ]] || { echo "missing checkpoint dir/base/manifest"; exit 2; }
mkdir -p "$OUTPUT_DIR"

declare -A checkpoints
for epoch in 2 4 6 8; do
  zero_based=$(printf '%02d' "$((epoch - 1))")
  matches=("$CHECKPOINT_DIR"/grpo-"$zero_based"-*.ckpt)
  [[ ${#matches[@]} -eq 1 && -f "${matches[0]}" ]] || { echo "expected one checkpoint for epoch $epoch"; exit 2; }
  checkpoints[$epoch]="${matches[0]}"
done

run_cell() {
  local checkpoint="$1" output="$2" noise="$3" gpu="$4"
  bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage19_eval.sh" \
    full "$checkpoint" "$output" "$MANIFEST" "$LIMIT" train "$gpu" "$noise" default default 2
}

pids=()
run_cell "$BASE_CHECKPOINT" "$OUTPUT_DIR/base_default.json" -1 0 & pids+=("$!")
run_cell "$BASE_CHECKPOINT" "$OUTPUT_DIR/base_ns20260729.json" 20260729 1 & pids+=("$!")
gpu=2
for epoch in 2 4 6; do
  run_cell "${checkpoints[$epoch]}" "$OUTPUT_DIR/epoch${epoch}_default.json" -1 "$gpu" & pids+=("$!"); gpu=$((gpu + 1))
  run_cell "${checkpoints[$epoch]}" "$OUTPUT_DIR/epoch${epoch}_ns20260729.json" 20260729 "$gpu" & pids+=("$!"); gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_cell "${checkpoints[8]}" "$OUTPUT_DIR/epoch8_default.json" -1 0 & pids+=("$!")
run_cell "${checkpoints[8]}" "$OUTPUT_DIR/epoch8_ns20260729.json" 20260729 1 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

select_args=(
  --manifest "$MANIFEST"
  --base -1 "$OUTPUT_DIR/base_default.json"
  --base 20260729 "$OUTPUT_DIR/base_ns20260729.json"
)
for epoch in 2 4 6 8; do
  sha="$($PYTHON_BIN -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(x) for x in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())' "${checkpoints[$epoch]}")"
  select_args+=(--candidate "$epoch" -1 "$OUTPUT_DIR/epoch${epoch}_default.json" "$sha")
  select_args+=(--candidate "$epoch" 20260729 "$OUTPUT_DIR/epoch${epoch}_ns20260729.json" "$sha")
done
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/select_grpo_stage21_development.py" \
  "${select_args[@]}" --output "$OUTPUT_DIR/selection.json"
