#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 internal-test|dev-select CANDIDATE_CHECKPOINT OUTPUT_DIR"
  exit 2
fi
GATE="$1"; CANDIDATE="$2"; OUTPUT_DIR="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
METRIC_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"

[[ "$GATE" == internal-test || "$GATE" == dev-select ]] || { echo "invalid gate"; exit 2; }
[[ -f "$CANDIDATE" && -f "$BASE_CHECKPOINT" && ! "$CANDIDATE" -ef "$BASE_CHECKPOINT" ]] || { echo "invalid candidate/base"; exit 2; }
mkdir -p "$OUTPUT_DIR"

if [[ "$GATE" == internal-test ]]; then
  MANIFEST="$ROOT_DIR/artifacts/grpo_stage21/manifests/internal_test_manifest.json"
  LIMIT=1023; LOG_SPLIT=train
  CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
  NOISES=(-1 20260729 20260730 20260731)
else
  MANIFEST="$ROOT_DIR/artifacts/grpo_stage0/dev_select3072_manifest.json"
  LIMIT=3072; LOG_SPLIT=val
  CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_dev4096_feature_cache_20260717"
  NOISES=(-1 20260729)
fi
[[ -f "$MANIFEST" && -d "$CACHE" && -d "$METRIC_CACHE" ]] || { echo "missing manifest/cache"; exit 2; }

pids=(); gpu=0
for noise in "${NOISES[@]}"; do
  bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage19_eval.sh" \
    full "$BASE_CHECKPOINT" "$OUTPUT_DIR/base_ns${noise}.json" "$MANIFEST" \
    "$LIMIT" "$LOG_SPLIT" "$gpu" "$noise" "$CACHE" "$METRIC_CACHE" 2 &
  pids+=("$!"); gpu=$((gpu + 1))
  bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage19_eval.sh" \
    full "$CANDIDATE" "$OUTPUT_DIR/candidate_ns${noise}.json" "$MANIFEST" \
    "$LIMIT" "$LOG_SPLIT" "$gpu" "$noise" "$CACHE" "$METRIC_CACHE" 2 &
  pids+=("$!"); gpu=$((gpu + 1))
done
for pid in "${pids[@]}"; do wait "$pid"; done

sha="$($PYTHON_BIN -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(x) for x in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())' "$CANDIDATE")"
gate_args=(--gate "$GATE" --manifest "$MANIFEST" --expected-candidate-sha256 "$sha")
for noise in "${NOISES[@]}"; do
  gate_args+=(--base "$noise" "$OUTPUT_DIR/base_ns${noise}.json")
  gate_args+=(--candidate "$noise" "$OUTPUT_DIR/candidate_ns${noise}.json")
done
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/check_grpo_stage21_transfer_gate.py" \
  "${gate_args[@]}" --output "$OUTPUT_DIR/gate.json"
