#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "Usage: $0 SELECTOR_CKPT STAGE21_EPOCH8_CKPT OUTPUT_DIR [GPU0 GPU1 GPU2]"
  exit 2
fi
SELECTOR="$1"; STAGE21="$2"; OUTPUT_DIR="$3"
GPU0="${4:-0}"; GPU1="${5:-1}"; GPU2="${6:-2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
FIT_MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
FOLD4_MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/fold4_manifest.json"

for path in "$SELECTOR" "$STAGE21" "$BASE_CHECKPOINT" "$FIT_MANIFEST" "$FOLD4_MANIFEST"; do
  [[ -f "$path" ]] || { echo "missing Stage23 gate input: $path"; exit 2; }
done
mkdir -p "$OUTPUT_DIR/calibration_raw" "$OUTPUT_DIR/diagnostic"

run_eval() {
  bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage23_selector_eval.sh" "$@"
}

pids=()
run_eval "$BASE_CHECKPOINT" "$SELECTOR" 0.0 0.5 "$FIT_MANIFEST" 4075 -1 \
  "$OUTPUT_DIR/calibration_raw/default.json" "$GPU0" & pids+=("$!")
run_eval "$BASE_CHECKPOINT" "$SELECTOR" 0.0 0.5 "$FIT_MANIFEST" 4075 20260811 \
  "$OUTPUT_DIR/calibration_raw/ns20260811.json" "$GPU1" & pids+=("$!")
run_eval "$BASE_CHECKPOINT" "$SELECTOR" 0.0 0.5 "$FIT_MANIFEST" 4075 20260812 \
  "$OUTPUT_DIR/calibration_raw/ns20260812.json" "$GPU2" & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/calibrate_grpo_stage23_selector.py" \
  --artifact "$OUTPUT_DIR/calibration_raw/default.json" \
  --artifact "$OUTPUT_DIR/calibration_raw/ns20260811.json" \
  --artifact "$OUTPUT_DIR/calibration_raw/ns20260812.json" \
  --output "$OUTPUT_DIR/calibration.json"

read -r MARGIN THRESHOLD < <(
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["residual_margin"],p["selected"]["threshold"])' \
    "$OUTPUT_DIR/calibration.json"
)
pids=()
run_eval "$STAGE21" "$SELECTOR" "$MARGIN" "$THRESHOLD" "$FOLD4_MANIFEST" 1021 -1 \
  "$OUTPUT_DIR/diagnostic/default.json" "$GPU0" & pids+=("$!")
run_eval "$STAGE21" "$SELECTOR" "$MARGIN" "$THRESHOLD" "$FOLD4_MANIFEST" 1021 20260729 \
  "$OUTPUT_DIR/diagnostic/ns20260729.json" "$GPU1" & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/check_grpo_stage23_selector_gate.py" \
  --artifact "$OUTPUT_DIR/diagnostic/default.json" \
  --artifact "$OUTPUT_DIR/diagnostic/ns20260729.json" \
  --calibration "$OUTPUT_DIR/calibration.json" \
  --output "$OUTPUT_DIR/diagnostic/gate.json"
