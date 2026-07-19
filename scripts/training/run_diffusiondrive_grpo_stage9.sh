#!/usr/bin/env bash
set -euo pipefail

# Registered selector-first stage-9 runner.  The only accepted objectives are
# selector_group and generation_group; generation is intentionally started by
# a separate invocation after selector gate failure.
if [[ $# -lt 3 || $# -gt 8 ]]; then
  echo "Usage: $0 selector_group|generation_group INPUT_CHECKPOINT EXPERIMENT_NAME [EPOCHS=1] [RESUME_CHECKPOINT=none] [SEED=0] [CUDA_DEVICE=0] [MAX_STEPS=0]"
  exit 2
fi
MODE="$1"
INPUT_CHECKPOINT="$2"
EXPERIMENT_NAME="$3"
EPOCHS="${4:-1}"
RESUME_CHECKPOINT="${5:-none}"
SEED="${6:-0}"
CUDA_DEVICE="${7:-0}"
MAX_STEPS="${8:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"

[[ "$MODE" == selector_group || "$MODE" == generation_group ]] || { echo "MODE must be selector_group or generation_group"; exit 2; }
[[ -f "$INPUT_CHECKPOINT" ]] || { echo "Input checkpoint does not exist: $INPUT_CHECKPOINT"; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Reference checkpoint does not exist: $BASE_CHECKPOINT"; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive"; exit 2; }
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "MAX_STEPS must be non-negative"; exit 2; }
if [[ "$RESUME_CHECKPOINT" != none && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT"; exit 2
fi

TRAIN_BATCH_LIMIT=3060
if [[ "$MAX_STEPS" != 0 ]]; then
  TRAIN_BATCH_LIMIT="$MAX_STEPS"
fi

ARGS=(
  navsim/planning/script/run_training.py
  agent=diffusiondrive_agent train_test_split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed="$SEED" dataloader.params.batch_size=2 dataloader.params.num_workers=4
  trainer.params.max_epochs="$EPOCHS" trainer.params.limit_train_batches="$TRAIN_BATCH_LIMIT"
  trainer.params.limit_val_batches=128 trainer.params.accelerator=gpu
  trainer.params.strategy=auto trainer.params.precision=32-true
  +trainer.params.devices=1 +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=16
  agent.checkpoint_path="$INPUT_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT"
  agent.lr=1e-6 agent.config.grpo_training_mode="$MODE"
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform
  agent.config.grpo_old_policy_sync_steps=32 agent.config.grpo_clip_ratio=0.2
  agent.config.selection_behavior_weighting=old_policy agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0 agent.config.selection_rank_loss_weight=0.0
  agent.config.grpo_priority_manifest_path= agent.config.grpo_priority_sample_fraction=0.0
  agent.config.generation_advantage_mode=group_zscore agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform agent.config.generation_trust_projection_mode=none
  agent.config.selector_consistency_kl_weight=0.0 agent.config.diffusion_truncation_timestep=8
  agent.config.diffusion_roll_timesteps=[8,0] agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1 agent.config.grpo_checkpoint_every_n_train_steps=128
  experiment_name="$EXPERIMENT_NAME"
)
if [[ "$MODE" == selector_group ]]; then
  ARGS+=(
    agent.config.policy_loss_weight=1.0 agent.config.kl_loss_weight=0.01
    agent.config.selector_generation_kl_weight=0.1
    agent.config.generation_policy_loss_weight=0.0
    agent.config.generation_kl_loss_weight=0.0
  )
else
  ARGS+=(
    agent.config.policy_loss_weight=0.0 agent.config.kl_loss_weight=0.0
    agent.config.selector_generation_kl_weight=0.0
    agent.config.generation_policy_loss_weight=1.0
    agent.config.generation_kl_loss_weight=0.1
  )
fi
if [[ "$MAX_STEPS" != 0 ]]; then
  ARGS+=("+trainer.params.max_steps=$MAX_STEPS")
fi
if [[ "$RESUME_CHECKPOINT" != none ]]; then ARGS+=("+resume_checkpoint_path=$RESUME_CHECKPOINT"); fi
mkdir -p "$ROOT_DIR/artifacts/grpo_stage9"
echo "stage9 mode=$MODE epochs=$EPOCHS max_steps=$MAX_STEPS resume=$RESUME_CHECKPOINT seed=$SEED"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "${ARGS[@]}"
