#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
SOURCE_DIR="$ROOT_DIR/artifacts/grpo_stage19/base_mode_source_shards"
DEFAULT_ARTIFACT="$ROOT_DIR/artifacts/grpo_stage19/base_full_navtrain6119.json"
OUTPUT_ROOT="$ROOT_DIR/artifacts/grpo_stage21"
SHARD_DIR="$OUTPUT_ROOT/base_ns20260728_shards"
MERGED="$OUTPUT_ROOT/base_full_navtrain6119_ns20260728.json"
MANIFEST_DIR="$OUTPUT_ROOT/manifests"
ALL_MANIFEST="$ROOT_DIR/artifacts/grpo_stage19/manifests/all_manifest.json"

[[ -f "$BASE_CHECKPOINT" && -f "$DEFAULT_ARTIFACT" && -f "$ALL_MANIFEST" ]] || {
  echo "missing registered base/default artifact/log manifest"; exit 2;
}
mkdir -p "$SHARD_DIR" "$MANIFEST_DIR"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  source_manifest="$SOURCE_DIR/shard0${gpu}.json"
  output="$SHARD_DIR/shard0${gpu}.json"
  [[ -f "$source_manifest" ]] || { echo "missing $source_manifest"; exit 2; }
  bash "$ROOT_DIR/scripts/evaluation/run_diffusiondrive_grpo_stage19_eval.sh" \
    full "$BASE_CHECKPOINT" "$output" "$source_manifest" \
    "$($PYTHON_BIN -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' "$source_manifest")" \
    train "$gpu" 20260728 default default 2 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/merge_grpo_stage17_artifacts.py" \
  --source-manifest "$ALL_MANIFEST" \
  --artifacts "$SHARD_DIR"/shard00.json "$SHARD_DIR"/shard01.json \
    "$SHARD_DIR"/shard02.json "$SHARD_DIR"/shard03.json \
    "$SHARD_DIR"/shard04.json "$SHARD_DIR"/shard05.json \
    "$SHARD_DIR"/shard06.json "$SHARD_DIR"/shard07.json \
  --output "$MERGED"

"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/build_grpo_stage21_manifests.py" \
  --default-artifact "$DEFAULT_ARTIFACT" --alternate-artifact "$MERGED" \
  --token-log-manifest "$ALL_MANIFEST" --output-dir "$MANIFEST_DIR" \
  --seed 20260728

