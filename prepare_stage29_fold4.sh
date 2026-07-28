#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
cd "$ROOT_DIR"
"$PYTHON_BIN" scripts/evaluation/freeze_grpo_stage29_fold4_inputs.py \
  --pilot-inputs artifacts/grpo_stage29/pilot/inputs.json \
  --control-freeze artifacts/grpo_stage28/pilot/training/A/formal/checkpoints.json \
  --control-audit artifacts/grpo_stage28/pilot/training/A/formal/audit.json \
  --branch H \
    artifacts/grpo_stage29/pilot/training/H/formal/checkpoints.json \
    artifacts/grpo_stage29/pilot/training/H/formal/audit.json \
  --branch HC \
    artifacts/grpo_stage29/pilot/training/HC/formal/checkpoints.json \
    artifacts/grpo_stage29/pilot/training/HC/formal/audit.json \
  --output artifacts/grpo_stage29/pilot/fold4/inputs.json
echo "PASS Stage29 fold4 inputs prepared"
echo "Next: bash ./run_stage29_fold4_8gpu.sh"
