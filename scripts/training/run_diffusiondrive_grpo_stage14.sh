#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 audit|formal SEED EXPERIMENT_NAME CUDA_DEVICE"
  exit 2
fi

PHASE="$1"
SEED="$2"
EXPERIMENT_NAME="$3"
CUDA_DEVICE="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/navsim/bin/python"
BASE_CHECKPOINT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model"
TRAINING_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"

[[ "$PHASE" == audit || "$PHASE" == formal ]] || {
  echo "PHASE must be audit or formal"; exit 2;
}
[[ "$SEED" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || {
  echo "CUDA_DEVICE must be an integer from 0 to 7"; exit 2;
}
[[ "$EXPERIMENT_NAME" =~ ^stage14_[A-Za-z0-9_.-]+$ ]] || {
  echo "EXPERIMENT_NAME must start with stage14_"; exit 2;
}
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Locked base checkpoint is missing"; exit 2; }
[[ -d "$TRAINING_CACHE" ]] || { echo "Locked training cache is missing"; exit 2; }

if [[ -n "${GRPO_BASE_CHECKPOINT:-}" && "$GRPO_BASE_CHECKPOINT" != "$BASE_CHECKPOINT" ]]; then
  echo "GRPO_BASE_CHECKPOINT must equal the locked Stage-14 base"
  exit 2
fi
if [[ -n "${GRPO_RESUME_CHECKPOINT:-}" ]]; then
  echo "Stage 14 forbids checkpoint resume"
  exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_DEVICE")"
[[ "$GPU_NAME" == *"RTX 4090"* ]] || {
  echo "Stage 14 requires RTX 4090; got: $GPU_NAME"; exit 2;
}

MAX_STEPS=8
CHECKPOINT_INTERVAL=0
if [[ "$PHASE" == formal ]]; then
  MAX_STEPS=128
  CHECKPOINT_INTERVAL=64
fi

echo "stage14 phase=$PHASE seed=$SEED max_steps=$MAX_STEPS fresh_base=$BASE_CHECKPOINT gpu=$CUDA_DEVICE ($GPU_NAME)"

cd "$ROOT_DIR"
TRAIN_ARGS=(
  navsim/planning/script/run_training.py
  agent=diffusiondrive_agent
  train_test_split=trainval
  cache_path="$TRAINING_CACHE"
  use_cache_without_dataset=true
  force_cache_computation=false
  seed="$SEED"
  dataloader.params.batch_size=2
  dataloader.params.num_workers=4
  trainer.params.max_epochs=4
  +trainer.params.max_steps="$MAX_STEPS"
  trainer.params.limit_train_batches="$MAX_STEPS"
  trainer.params.limit_val_batches="$MAX_STEPS"
  trainer.params.accumulate_grad_batches=1
  trainer.params.accelerator=gpu
  trainer.params.strategy=auto
  trainer.params.precision=32-true
  +trainer.params.devices=1
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$BASE_CHECKPOINT"
  agent.reference_checkpoint_path="$BASE_CHECKPOINT"
  agent.lr=1e-6
  agent.config.grpo_training_mode=generation
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_INTERVAL"
  agent.config.grpo_old_policy_sync_steps=32
  agent.config.policy_loss_weight=0.0
  agent.config.kl_loss_weight=0.0
  agent.config.selector_generation_kl_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0
  agent.config.grpo_reward_mode=pdms
  agent.config.grpo_scene_weight_mode=uniform
  agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.1
  agent.config.generation_adaptive_kl_enabled=false
  agent.config.generation_advantage_mode=collision_truncated_intra_anchor
  agent.config.grpo_rollouts_per_mode=2
  agent.config.grpo_priority_manifest_path=""
  agent.config.grpo_priority_sample_fraction=0.0
  agent.config.generation_mode_weighting=uniform
  agent.config.generation_ddim_eta=1.0
  agent.config.generation_final_std=0.05
  agent.config.generation_trust_projection_mode=none
  agent.config.generation_trust_calibration_path=""
  agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0
  agent.config.diffusion_truncation_timestep=8
  agent.config.diffusion_roll_timesteps=[8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  experiment_name="$EXPERIMENT_NAME"
)

GRPO_BASE_CHECKPOINT="$BASE_CHECKPOINT" \
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
