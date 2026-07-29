#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/evaluation/summarize_grpo_stage37_generator.py" \
 --stage37-root "$ROOT_DIR/artifacts/grpo_stage37" \
 --baseline-root "$ROOT_DIR/artifacts/grpo_stage34/pilot/eval" \
 --stage36-root "$ROOT_DIR/artifacts/grpo_stage36" \
 --bucket-manifest "$ROOT_DIR/artifacts/grpo_stage30/manifests/buckets_6119.json" \
 --output "$ROOT_DIR/artifacts/grpo_stage37/pilot/generator_summary.json" \
 --selection-output "$ROOT_DIR/artifacts/grpo_stage37/pilot/generator_selection.json"
[[ -f "$ROOT_DIR/artifacts/grpo_stage37/pilot/generator_selection.json" ]] || { echo "Stage37 generator gate did not pass"; exit 1; }
echo "PASS Stage37 generator gate; selection=artifacts/grpo_stage37/pilot/generator_selection.json"
