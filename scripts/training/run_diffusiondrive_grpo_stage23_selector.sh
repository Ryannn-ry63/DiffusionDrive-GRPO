#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 audit|formal EXPERIMENT_NAME CUDA_DEVICE"
  exit 2
fi
PHASE="$1"; EXPERIMENT_NAME="$2"; CUDA_DEVICE="$3"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"
BANK_DIR="${STAGE23_BANK_DIR:-$ROOT_DIR/artifacts/grpo_stage23/candidate_bank}"
BANK_DEFAULT="$BANK_DIR/base_default.json"
BANK_NS1="$BANK_DIR/base_ns20260811.json"
BANK_NS2="$BANK_DIR/base_ns20260812.json"

[[ "$PHASE" == audit || "$PHASE" == formal ]] || { echo "invalid phase"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage23_selector_[A-Za-z0-9_.-]+$ ]] || { echo "invalid experiment name"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "invalid CUDA device"; exit 2; }
for path in "$BASE_CHECKPOINT" "$MANIFEST" "$BANK_DEFAULT" "$BANK_NS1" "$BANK_NS2"; do
  [[ -f "$path" ]] || { echo "missing locked Stage23 input: $path"; exit 2; }
done

MAX_EPOCHS=8
LIMIT_TRAIN=1.0
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_TRAIN=8
  EXTRA_ARGS+=(+trainer.params.max_steps=8)
fi

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent train_test_split=trainval \
  cache_path="$TRAINING_CACHE" use_cache_without_dataset=true force_cache_computation=false \
  seed=0 dataloader.params.batch_size=2 dataloader.params.num_workers=4 \
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.limit_train_batches="$LIMIT_TRAIN" \
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=1 \
  trainer.params.accelerator=gpu trainer.params.strategy=auto trainer.params.precision=32-true \
  +trainer.params.devices=1 +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1 \
  agent.checkpoint_path="$BASE_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT" \
  agent.lr=1e-4 agent.config.grpo_training_mode=stage23_selector \
  agent.config.stage23_selector_train_manifest_path="$MANIFEST" \
  "agent.config.stage23_candidate_bank_paths=[$BANK_DEFAULT,$BANK_NS1,$BANK_NS2]" \
  agent.config.stage23_selector_num_members=5 agent.config.stage23_selector_num_modes=20 \
  agent.config.stage23_selector_dim=128 \
  agent.config.stage23_selector_pair_reward_gap=0.01 \
  agent.config.stage23_selector_focal_gamma=2.0 \
  agent.config.stage23_selector_unsafe_positive_weight=50.0 \
  agent.config.grpo_checkpoint_save_top_k=-1 \
  agent.config.diffusion_truncation_timestep=32 \
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0] \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  experiment_name="$EXPERIMENT_NAME" "${EXTRA_ARGS[@]}"
