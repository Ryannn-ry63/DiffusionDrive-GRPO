#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "Usage: $0 audit|development|formal INPUT_CHECKPOINT EXPERIMENT_NAME EPOCHS [RESUME_CHECKPOINT=none] [SEED=0] [CUDA_DEVICES=0] [MAX_STEPS=0]"
  exit 2
fi

PHASE="$1"
INPUT_CHECKPOINT="$2"
EXPERIMENT_NAME="$3"
EPOCHS="$4"
RESUME_CHECKPOINT="${5:-none}"
SEED="${6:-0}"
CUDA_DEVICES="${7:-0}"
MAX_STEPS="${8:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${GRPO_TRAIN_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
TRAIN_MANIFEST="$ROOT_DIR/artifacts/grpo_stage15/manifests/selector_train_manifest.json"

[[ "$PHASE" == audit || "$PHASE" == development || "$PHASE" == formal ]] || { echo "PHASE must be audit, development, or formal"; exit 2; }
[[ -f "$INPUT_CHECKPOINT" && -f "$BASE_CHECKPOINT" ]] || { echo "input or base checkpoint missing"; exit 2; }
[[ -d "$TRAINING_CACHE" && -f "$TRAIN_MANIFEST" ]] || { echo "cache or Stage16 manifest missing"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage16_[A-Za-z0-9_.-]+$ ]] || { echo "EXPERIMENT_NAME must start with stage16_"; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive"; exit 2; }
[[ "$SEED" =~ ^[0-2]$ ]] || { echo "SEED must be 0, 1, or 2"; exit 2; }
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "MAX_STEPS must be non-negative"; exit 2; }
if [[ "$RESUME_CHECKPOINT" != none && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "resume checkpoint missing: $RESUME_CHECKPOINT"; exit 2
fi
if [[ "$PHASE" == audit && "$RESUME_CHECKPOINT" != none && "$MAX_STEPS" == 32 ]] && (( EPOCHS < 2 )); then
  echo "U8-to-U32 resume audit requires EPOCHS>=2 so Lightning can enter the resumed epoch"; exit 2
fi

DEVICES=1
BATCH_SIZE=2
ACCUMULATE=1
STRATEGY=auto
MANIFEST_ARG="$TRAIN_MANIFEST"
if [[ "$PHASE" == audit ]]; then
  [[ "$CUDA_DEVICES" =~ ^[0-7]$ ]] || { echo "audit requires one CUDA device"; exit 2; }
  [[ "$MAX_STEPS" == 8 || "$MAX_STEPS" == 32 ]] || { echo "audit MAX_STEPS must be 8 or 32"; exit 2; }
elif [[ "$PHASE" == development || "$PHASE" == formal ]]; then
  [[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "$PHASE requires eight comma-separated CUDA devices"; exit 2; }
  DEVICES=8
  BATCH_SIZE=1
  ACCUMULATE=8
  STRATEGY=ddp
  [[ "$MAX_STEPS" == 0 ]] || { echo "$PHASE does not accept MAX_STEPS"; exit 2; }
  if [[ "$PHASE" == development ]]; then
    [[ "$EPOCHS" == 10 ]] || { echo "development is locked to 10 epochs"; exit 2; }
  else
    MANIFEST_ARG=""
  fi
fi

EXTRA_ARGS=()
if [[ "$MAX_STEPS" != 0 ]]; then EXTRA_ARGS+=(+trainer.params.max_steps="$MAX_STEPS"); fi
if [[ "$RESUME_CHECKPOINT" != none ]]; then EXTRA_ARGS+=(+resume_checkpoint_path="$RESUME_CHECKPOINT"); fi

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent train_test_split=trainval \
  cache_path="$TRAINING_CACHE" use_cache_without_dataset=true force_cache_computation=false \
  seed="$SEED" dataloader.params.batch_size="$BATCH_SIZE" dataloader.params.num_workers=4 \
  trainer.params.max_epochs="$EPOCHS" trainer.params.limit_train_batches=1.0 \
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches="$ACCUMULATE" \
  trainer.params.accelerator=gpu trainer.params.strategy="$STRATEGY" trainer.params.precision=32-true \
  +trainer.params.devices="$DEVICES" +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1 \
  agent.checkpoint_path="$INPUT_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT" \
  agent.lr=1e-6 agent.config.grpo_training_mode=diffgrpo_full_chain \
  agent.config.grpo_decoder_gradient_scope=all_layers \
  agent.config.inference_selector_source=reference \
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform \
  agent.config.selection_behavior_weighting=old_policy \
  agent.config.selection_entropy_weight=0.0 agent.config.selection_exploration_floor=0.0 \
  agent.config.selection_rank_loss_weight=0.0 agent.config.selector_consistency_kl_weight=0.0 \
  agent.config.selector_generation_kl_weight=0.0 \
  agent.config.policy_loss_weight=0.0 agent.config.kl_loss_weight=0.0 \
  agent.config.generation_policy_loss_weight=1.0 agent.config.generation_kl_loss_weight=0.0 \
  agent.config.generation_adaptive_kl_enabled=false \
  agent.config.generation_policy_algorithm=diffgrpo_full_chain \
  agent.config.diffgrpo_bc_weight=0.1 agent.config.diffgrpo_step_discount=0.6 \
  agent.config.diffgrpo_logprob_reduction=mean \
  agent.config.diffgrpo_train_manifest_path="$MANIFEST_ARG" \
  agent.config.generation_advantage_mode=group_zscore agent.config.grpo_rollouts_per_mode=1 \
  agent.config.generation_mode_weighting=uniform agent.config.generation_trust_projection_mode=none \
  agent.config.grpo_old_policy_sync_steps=32 \
  agent.config.grpo_priority_manifest_path= agent.config.grpo_priority_sample_fraction=0.0 \
  agent.config.diffusion_truncation_timestep=32 \
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0] \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  agent.config.grpo_checkpoint_save_top_k=-1 \
  experiment_name="$EXPERIMENT_NAME" "${EXTRA_ARGS[@]}"
