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
MANIFEST="$ROOT_DIR/artifacts/grpo_stage22/manifests/inner_calibration_manifest.json"
[[ -d "$CHECKPOINT_DIR" && -f "$BASE_CHECKPOINT" && -f "$MANIFEST" ]] || { echo "missing Stage22 calibration input"; exit 2; }
mkdir -p "$OUTPUT_DIR"

declare -A merged
for epoch in 1 2 3 4; do
  zero_based="$(printf '%02d' "$((epoch - 1))")"
  matches=("$CHECKPOINT_DIR"/grpo-"$zero_based"-*.ckpt)
  [[ ${#matches[@]} -eq 1 && -f "${matches[0]}" ]] || { echo "expected one Stage22 checkpoint for epoch $epoch"; exit 2; }
  merged[$epoch]="$OUTPUT_DIR/epoch${epoch}_merged.ckpt"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/merge_grpo_stage22_lora.py" --input "${matches[0]}" --output "${merged[$epoch]}" --audit-output "$OUTPUT_DIR/epoch${epoch}_merge.json"
done

run_cell() {
  local checkpoint="$1" output="$2" noise="$3" gpu="$4"
  bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage22_eval.sh" full "$checkpoint" "$output" "$MANIFEST" 1019 train "$gpu" "$noise"
}
pids=()
run_cell "$BASE_CHECKPOINT" "$OUTPUT_DIR/base_ns-1.json" -1 0 & pids+=("$!")
run_cell "$BASE_CHECKPOINT" "$OUTPUT_DIR/base_ns20260801.json" 20260801 1 & pids+=("$!")
gpu=2
for epoch in 1 2 3; do
  run_cell "${merged[$epoch]}" "$OUTPUT_DIR/epoch${epoch}_ns-1.json" -1 "$gpu" & pids+=("$!"); gpu=$((gpu + 1))
  run_cell "${merged[$epoch]}" "$OUTPUT_DIR/epoch${epoch}_ns20260801.json" 20260801 "$gpu" & pids+=("$!"); gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do wait "$pid"; done
pids=()
run_cell "${merged[4]}" "$OUTPUT_DIR/epoch4_ns-1.json" -1 0 & pids+=("$!")
run_cell "${merged[4]}" "$OUTPUT_DIR/epoch4_ns20260801.json" 20260801 1 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

gate_args=(--gate inner --manifest "$MANIFEST" --base -1 "$OUTPUT_DIR/base_ns-1.json" --base 20260801 "$OUTPUT_DIR/base_ns20260801.json")
for epoch in 1 2 3 4; do
  sha="$("$PYTHON_BIN" -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(x) for x in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())' "${merged[$epoch]}")"
  gate_args+=(--candidate "$epoch" -1 "$OUTPUT_DIR/epoch${epoch}_ns-1.json" "$sha")
  gate_args+=(--candidate "$epoch" 20260801 "$OUTPUT_DIR/epoch${epoch}_ns20260801.json" "$sha")
done
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/check_grpo_stage22_gate.py" "${gate_args[@]}" --output "$OUTPUT_DIR/selection.json"
