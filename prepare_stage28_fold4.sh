#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
cd "$ROOT_DIR"
"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage28_fold4_inputs.py \
  --pilot-inputs artifacts/grpo_stage28/pilot/inputs.json \
  --branch A \
    artifacts/grpo_stage28/pilot/training/A/formal/checkpoints.json \
    artifacts/grpo_stage28/pilot/training/A/formal/audit.json \
  --branch B \
    artifacts/grpo_stage28/pilot/training/B/formal/checkpoints.json \
    artifacts/grpo_stage28/pilot/training/B/formal/audit.json \
  --branch C \
    artifacts/grpo_stage28/pilot/training/C/formal/checkpoints.json \
    artifacts/grpo_stage28/pilot/training/C/formal/audit.json \
  --output artifacts/grpo_stage28/pilot/fold4/inputs.json
echo "PASS Stage28 fold4 inputs prepared"
echo "Next: bash ./run_stage28_fold4_8gpu.sh"
