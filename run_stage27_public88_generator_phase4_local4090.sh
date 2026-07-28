#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export STAGE27_EXPECTED_GPU_SUBSTRING="GeForce RTX 4090"
exec bash "$ROOT_DIR/run_stage27_public88_generator_phase4_h100.sh"
