#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 MANIFEST LIMIT DOMAIN GENERATOR NAMESPACE OUTPUT GPU"
  exit 2
fi
MANIFEST="$1"; LIMIT="$2"; DOMAIN="$3"; GENERATOR="$4"
NAMESPACE="$5"; OUTPUT="$6"; GPU="$7"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
METRIC_CACHE="${NAVSIM_METRIC_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval}"

for path in "$MANIFEST" "$GENERATOR" "$BASE_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "missing Stage24 input: $path"; exit 2; }
done
[[ "$LIMIT" =~ ^[1-9][0-9]*$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid limit/GPU"; exit 2; }
[[ "$DOMAIN" =~ ^[a-z0-9_]+$ ]] || { echo "invalid generator domain"; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
  "$ROOT_DIR/scripts/evaluation/evaluate_grpo_schedule.py" \
  --checkpoint "$GENERATOR" --reference-checkpoint "$BASE_CHECKPOINT" \
  --cache-path "$TRAINING_CACHE" --metric-cache-path "$METRIC_CACHE" \
  --tokens-file "$MANIFEST" --limit "$LIMIT" --log-split train \
  --batch-size 2 --num-workers 4 --device cuda:0 \
  --truncation-timestep 32 --roll-timesteps 32 24 16 8 0 \
  --scheduler-num-inference-steps 125 --generation-policy-algorithm legacy_ppo \
  --evaluation-noise-namespace "$NAMESPACE" --selector-logits-source current \
  --generator-domain "$DOMAIN" --store-candidate-trajectories --output "$OUTPUT"

"$PYTHON_BIN" -c '
import json,sys
p=json.load(open(sys.argv[1])); s=p["summary"]
assert s["completed"] and s["num_tokens"] == int(sys.argv[2])
assert s["generator_domain"] == sys.argv[3]
assert s["stores_candidate_trajectories"]
assert all(len(r["candidate_trajectories"]) == 20 for r in p["records"])
' "$OUTPUT" "$LIMIT" "$DOMAIN"
