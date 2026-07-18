#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 12 ]]; then
  echo "Usage: $0 G|S INPUT_CHECKPOINT EXPERIMENT_NAME UPDATES [EPSILON=0] [SCENE_WEIGHT_MODE=reference_headroom] [SEED=0] [CUDA_DEVICE=0] [BEHAVIOR_WEIGHTING=old_policy] [POLICY_WEIGHT=1] [RANK_WEIGHT=0] [LR=1e-6]"
  exit 2
fi

STAGE="$1"
INPUT_CHECKPOINT="$2"
EXPERIMENT_NAME="$3"
UPDATES="$4"
EPSILON="${5:-0}"
SCENE_WEIGHT_MODE="${6:-reference_headroom}"
SEED="${7:-0}"
CUDA_DEVICE="${8:-0}"
BEHAVIOR_WEIGHTING="${9:-old_policy}"
POLICY_WEIGHT="${10:-1.0}"
RANK_WEIGHT="${11:-0.0}"
LEARNING_RATE="${12:-1e-6}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
VAL_BATCHES="${GRPO_VAL_BATCHES:-128}"

if [[ ! -f "$INPUT_CHECKPOINT" ]]; then
  echo "Input checkpoint does not exist: $INPUT_CHECKPOINT"
  exit 2
fi
if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "Fixed reference checkpoint does not exist: $BASE_CHECKPOINT"
  exit 2
fi
if [[ "$SCENE_WEIGHT_MODE" != "uniform" && "$SCENE_WEIGHT_MODE" != "reference_headroom" ]]; then
  echo "SCENE_WEIGHT_MODE must be uniform or reference_headroom"
  exit 2
fi
if [[ "$BEHAVIOR_WEIGHTING" != "old_policy" && "$BEHAVIOR_WEIGHTING" != "uniform_valid" ]]; then
  echo "BEHAVIOR_WEIGHTING must be old_policy or uniform_valid"
  exit 2
fi
if ! [[ "$UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "UPDATES must be a positive integer"
  exit 2
fi

case "$STAGE" in
  G)
    TRAINING_MODE=generation
    EPSILON=0
    CATEGORICAL_KL=0.0
    GENERATION_KL=0.1
    RANK_WEIGHT=0.0
    ;;
  S)
    TRAINING_MODE=selector
    CATEGORICAL_KL=0.01
    GENERATION_KL=0.0
    ;;
  *)
    echo "STAGE must be G or S"
    exit 2
    ;;
esac

cd "$ROOT_DIR"
TRAIN_ARGS=(
  navsim/planning/script/run_training.py
  agent=diffusiondrive_agent
  train_test_split=trainval
  cache_path="$TRAIN_CACHE"
  use_cache_without_dataset=true
  force_cache_computation=false
  seed="$SEED"
  dataloader.params.batch_size=2
  dataloader.params.num_workers=4
  trainer.params.max_epochs=1
  trainer.params.limit_train_batches="$UPDATES"
  trainer.params.limit_val_batches="$VAL_BATCHES"
  trainer.params.accelerator=gpu
  trainer.params.strategy=auto
  trainer.params.precision=32-true
  +trainer.params.devices=1
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=16
  agent.checkpoint_path="$INPUT_CHECKPOINT"
  agent.reference_checkpoint_path="$BASE_CHECKPOINT"
  agent.lr="$LEARNING_RATE"
  agent.config.grpo_training_mode="$TRAINING_MODE"
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_reward_mode=pdms
  agent.config.grpo_scene_weight_mode="$SCENE_WEIGHT_MODE"
  agent.config.grpo_reference_gate_margin=0.01
  agent.config.grpo_reference_gate_scale=0.05
  agent.config.selection_exploration_floor="$EPSILON"
  agent.config.selection_behavior_weighting="$BEHAVIOR_WEIGHTING"
  agent.config.policy_loss_weight="$POLICY_WEIGHT"
  agent.config.selection_rank_loss_weight="$RANK_WEIGHT"
  agent.config.kl_loss_weight="$CATEGORICAL_KL"
  agent.config.selection_entropy_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0
  agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight="$GENERATION_KL"
  agent.config.generation_mode_weighting=uniform
  agent.config.generation_ddim_eta=1.0
  agent.config.generation_final_std=0.05
  agent.config.grpo_old_policy_sync_steps=32
  agent.config.diffusion_truncation_timestep=8
  agent.config.diffusion_roll_timesteps=[8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  experiment_name="$EXPERIMENT_NAME"
)

echo "stage=$STAGE input=$INPUT_CHECKPOINT reference=$BASE_CHECKPOINT updates=$UPDATES epsilon=$EPSILON behavior_weighting=$BEHAVIOR_WEIGHTING policy_weight=$POLICY_WEIGHT rank_weight=$RANK_WEIGHT lr=$LEARNING_RATE scene_weight=$SCENE_WEIGHT_MODE seed=$SEED"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
