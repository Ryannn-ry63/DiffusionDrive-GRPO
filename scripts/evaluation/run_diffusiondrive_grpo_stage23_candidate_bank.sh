#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIR"
  exit 2
fi
OUTPUT_DIR="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
METRIC_CACHE="${NAVSIM_METRIC_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval}"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
LIMIT=4075

[[ -f "$BASE_CHECKPOINT" && -f "$MANIFEST" ]] || { echo "missing base checkpoint/manifest"; exit 2; }
[[ -d "$TRAINING_CACHE" && -d "$METRIC_CACHE" ]] || { echo "missing NAVSIM cache"; exit 2; }
mkdir -p "$OUTPUT_DIR/shards"

run_full_namespace() {
  local namespace="$1" gpu="$2" output="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
    --checkpoint "$BASE_CHECKPOINT" --reference-checkpoint "$BASE_CHECKPOINT" \
    --cache-path "$TRAINING_CACHE" --metric-cache-path "$METRIC_CACHE" \
    --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train \
    --batch-size 2 --num-workers 4 --device cuda:0 \
    --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
    --scheduler-num-inference-steps 125 --generation-policy-algorithm legacy_ppo \
    --evaluation-noise-namespace "$namespace" --selector-logits-source current \
    --store-candidate-trajectories --output "$output"
}

if [[ "${STAGE23_BANK_STRATEGY:-full}" == "full" ]]; then
  pids=()
  run_full_namespace -1 0 "$OUTPUT_DIR/base_default.json" & pids+=("$!")
  run_full_namespace 20260811 1 "$OUTPUT_DIR/base_ns20260811.json" & pids+=("$!")
  run_full_namespace 20260812 2 "$OUTPUT_DIR/base_ns20260812.json" & pids+=("$!")
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [[ "$status" -eq 0 ]] || { echo "Stage23 full-bank worker failed"; exit 1; }
else
  [[ "${STAGE23_BANK_STRATEGY}" == "sharded" ]] || { echo "invalid bank strategy"; exit 2; }
fi

run_shard() {
  local namespace="$1" gpu="$2" offset="$3" count="$4" output="$5"
  if [[ -f "$output" ]] && "$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); s=p["summary"]
assert s["completed"] and s["num_tokens"] == int(sys.argv[2]) and s["token_offset"] == int(sys.argv[3])
' "$output" "$count" "$offset" 2>/dev/null; then
    echo "reuse completed shard: $output"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
    --checkpoint "$BASE_CHECKPOINT" --reference-checkpoint "$BASE_CHECKPOINT" \
    --cache-path "$TRAINING_CACHE" --metric-cache-path "$METRIC_CACHE" \
    --tokens-file "$MANIFEST" --token-offset "$offset" --limit "$count" --log-split train \
    --batch-size 2 --num-workers 4 --device cuda:0 \
    --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
    --scheduler-num-inference-steps 125 --generation-policy-algorithm legacy_ppo \
    --evaluation-noise-namespace "$namespace" --selector-logits-source current \
    --store-candidate-trajectories --output "$output"
}

run_namespace() {
  local namespace="$1" label="$2"
  local pids=() shard_paths=() status=0
  for shard in 0 1 2 3 4 5 6 7; do
    local offset=$((shard * 510)) count=510
    if [[ "$shard" -eq 7 ]]; then count=505; fi
    local output="$OUTPUT_DIR/shards/${label}_shard${shard}.json"
    shard_paths+=("$output")
    run_shard "$namespace" "$shard" "$offset" "$count" "$output" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [[ "$status" -eq 0 ]] || { echo "Stage23 namespace $namespace has failed shards"; exit 1; }
  local merge_args=()
  for path in "${shard_paths[@]}"; do merge_args+=(--shard "$path"); done
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/merge_grpo_stage23_candidate_bank.py" \
    --manifest "$MANIFEST" "${merge_args[@]}" --namespace "$namespace" \
    --output "$OUTPUT_DIR/${label}.json"
}

if [[ "${STAGE23_BANK_STRATEGY:-full}" == "sharded" ]]; then
  run_namespace -1 base_default
  run_namespace 20260811 base_ns20260811
  run_namespace 20260812 base_ns20260812
fi

"$PYTHON_BIN" -c '
import json, pathlib, sys
expected = {-1, 20260811, 20260812}
paths = [pathlib.Path(value) for value in sys.argv[1:]]
namespaces = set()
for path in paths:
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    assert summary["completed"] and summary["num_tokens"] == 4075
    assert summary["stores_candidate_trajectories"]
    assert all(len(record["candidate_trajectories"]) == 20 for record in payload["records"])
    namespaces.add(summary["evaluation_noise_namespace"])
assert namespaces == expected
print("Stage23 candidate bank complete and shape-audited")
' "$OUTPUT_DIR/base_default.json" "$OUTPUT_DIR/base_ns20260811.json" "$OUTPUT_DIR/base_ns20260812.json"
