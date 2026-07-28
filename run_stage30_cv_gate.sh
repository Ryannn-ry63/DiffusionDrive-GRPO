#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/navsim:${PYTHONPATH:-}"
cd "$ROOT_DIR"
"$PYTHON_BIN" scripts/evaluation/check_grpo_stage30_cv.py \
  --root "$ROOT_DIR" --output "$ROOT_DIR/artifacts/grpo_stage30/cv/report.json"
echo "PASS Stage30 four-fold OOF gate"
echo "Report: $ROOT_DIR/artifacts/grpo_stage30/cv/report.json"

