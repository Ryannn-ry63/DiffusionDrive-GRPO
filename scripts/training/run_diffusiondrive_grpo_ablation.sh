#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 14 ]]; then
  echo "Usage: $0 D|DPLUS|DPLUSS|DDENSE|DWEIGHT SEED [SELECTOR_KL] [EPOCHS] [CUDA_DEVICE] [GENERATION_KL] [PDM_EPSILON] [DENSE_WEIGHT] [MODE_WEIGHTING] [BATCH_SIZE] [ADVANTAGE_MODE] [PRIORITY_MANIFEST=none] [UPDATES=128] [ROLLOUTS_PER_MODE=1]"
  exit 2
fi

VARIANT="$1"
SEED="$2"
SELECTOR_KL="${3:-0.05}"
EPOCHS="${4:-4}"
CUDA_DEVICE="${5:-0}"
GENERATION_KL="${6:-0.1}"
PDM_EPSILON="${7:-0.001}"
DENSE_WEIGHT="${8:-0.1}"
MODE_WEIGHTING="${9:-uniform}"
BATCH_SIZE="${10:-1}"
ADVANTAGE_MODE="${11:-group_zscore}"
PRIORITY_MANIFEST="${12:-none}"
UPDATES="${13:-128}"
ROLLOUTS_PER_MODE="${14:-1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GENERATION_TRUST_PROJECTION_MODE="${GENERATION_TRUST_PROJECTION_MODE:-none}"
GENERATION_TRUST_CALIBRATION_PATH="${GENERATION_TRUST_CALIBRATION_PATH:-}"
if [[ "$GENERATION_TRUST_PROJECTION_MODE" != "none" && "$GENERATION_TRUST_PROJECTION_MODE" != "reference_mean_ball" ]]; then
  echo "GENERATION_TRUST_PROJECTION_MODE must be none or reference_mean_ball"
  exit 2
fi
if [[ "$GENERATION_TRUST_PROJECTION_MODE" == "reference_mean_ball" && ! -f "$GENERATION_TRUST_CALIBRATION_PATH" ]]; then
  echo "Immutable generation trust calibration does not exist: $GENERATION_TRUST_CALIBRATION_PATH"
  exit 2
fi

case "$VARIANT" in
  D)
    SELECTOR_KL=0.0
    REWARD_MODE=pdms
    ;;
  DPLUS)
    REWARD_MODE=pdms
    ;;
  DPLUSS)
    REWARD_MODE=pdm_tiebreak
    ;;
  DDENSE)
    REWARD_MODE=pdm_dense
    ;;
  DWEIGHT)
    REWARD_MODE=pdms
    ;;
  *)
    echo "Unknown variant: $VARIANT"
    exit 2
    ;;
esac

if [[ "$ADVANTAGE_MODE" != "group_zscore" && "$ADVANTAGE_MODE" != "reference_centered" && "$ADVANTAGE_MODE" != "within_anchor" && "$ADVANTAGE_MODE" != "hierarchical" && "$ADVANTAGE_MODE" != "anchor_hierarchical" && "$ADVANTAGE_MODE" != "anchor_rloo" ]]; then
  echo "ADVANTAGE_MODE must be group_zscore, reference_centered, within_anchor, hierarchical, anchor_hierarchical, or anchor_rloo"
  exit 2
fi
if [[ ( "$ADVANTAGE_MODE" == "within_anchor" || "$ADVANTAGE_MODE" == "hierarchical" ) && "$ROLLOUTS_PER_MODE" != "2" ]]; then
  echo "within_anchor/hierarchical require ROLLOUTS_PER_MODE=2"
  exit 2
fi
if [[ ( "$ADVANTAGE_MODE" == "group_zscore" || "$ADVANTAGE_MODE" == "reference_centered" ) && "$ROLLOUTS_PER_MODE" != "1" ]]; then
  echo "group_zscore/reference_centered require ROLLOUTS_PER_MODE=1"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_hierarchical" && "$ROLLOUTS_PER_MODE" != "2" && "$ROLLOUTS_PER_MODE" != "4" ]]; then
  echo "anchor_hierarchical requires ROLLOUTS_PER_MODE=2 or 4"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_rloo" && "$ROLLOUTS_PER_MODE" != "2" ]]; then
  echo "anchor_rloo requires ROLLOUTS_PER_MODE=2"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_rloo" && "$VARIANT" != "D" ]]; then
  echo "anchor_rloo is preregistered for raw PDMS (variant D) only"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_rloo" && "$BATCH_SIZE" != "2" ]]; then
  echo "anchor_rloo is preregistered with BATCH_SIZE=2"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_rloo" && "$MODE_WEIGHTING" != "uniform" ]]; then
  echo "anchor_rloo is preregistered with uniform mode weighting"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_rloo" && "$PRIORITY_MANIFEST" != "none" ]]; then
  echo "anchor_rloo is preregistered with uniform sampling"
  exit 2
fi
if [[ "$ADVANTAGE_MODE" == "anchor_rloo" && "$GENERATION_KL" != "0.1" ]]; then
  echo "anchor_rloo is preregistered with GENERATION_KL=0.1"
  exit 2
fi

ACCUMULATE_GRAD_BATCHES=1
FORWARD_BATCHES="$UPDATES"
OLD_POLICY_SYNC_STEPS=32
if [[ "$ROLLOUTS_PER_MODE" == "4" ]]; then
  if [[ "$BATCH_SIZE" != "1" ]]; then
    echo "K4 is preregistered with BATCH_SIZE=1 and gradient accumulation=2"
    exit 2
  fi
  ACCUMULATE_GRAD_BATCHES=2
  FORWARD_BATCHES=$((UPDATES * ACCUMULATE_GRAD_BATCHES))
  OLD_POLICY_SYNC_STEPS=64
fi

PRIORITY_FRACTION=0.0
PRIORITY_CONFIG_PATH=""
PRIORITY_TAG=uniform
if [[ "$PRIORITY_MANIFEST" != "none" ]]; then
  if [[ ! -f "$PRIORITY_MANIFEST" ]]; then
    echo "Priority manifest does not exist: $PRIORITY_MANIFEST"
    exit 2
  fi
  PRIORITY_FRACTION=0.5
  PRIORITY_CONFIG_PATH="$PRIORITY_MANIFEST"
  PRIORITY_TAG=priority50
fi
EXPERIMENT_NAME="grpo_${VARIANT,,}_seed${SEED}_skl${SELECTOR_KL}_gkl${GENERATION_KL}_eps${PDM_EPSILON}_dense${DENSE_WEIGHT}_mweight${MODE_WEIGHTING}_bs${BATCH_SIZE}_adv${ADVANTAGE_MODE}_k${ROLLOUTS_PER_MODE}_${PRIORITY_TAG}_u${UPDATES}"
cd "$ROOT_DIR"

TRAIN_ARGS=(
  navsim/planning/script/run_training.py
  agent=diffusiondrive_agent
  train_test_split=trainval
  cache_path=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache
  use_cache_without_dataset=true
  force_cache_computation=false
  seed="$SEED"
  dataloader.params.batch_size="$BATCH_SIZE"
  dataloader.params.num_workers=4
  trainer.params.max_epochs="$EPOCHS"
  +trainer.params.max_steps="$UPDATES"
  trainer.params.limit_train_batches="$FORWARD_BATCHES"
  trainer.params.limit_val_batches="$FORWARD_BATCHES"
  trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES"
  trainer.params.accelerator=gpu
  trainer.params.strategy=auto
  trainer.params.precision=32-true
  +trainer.params.devices=1
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=16
  agent.lr=1e-6
  agent.config.grpo_training_mode=generation
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps=64
  agent.config.grpo_old_policy_sync_steps="$OLD_POLICY_SYNC_STEPS"
  agent.config.selector_consistency_kl_weight="$SELECTOR_KL"
  agent.config.grpo_reward_mode="$REWARD_MODE"
  agent.config.pdm_tiebreak_max_epsilon="$PDM_EPSILON"
  agent.config.pdm_dense_weight="$DENSE_WEIGHT"
  agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight="$GENERATION_KL"
  agent.config.generation_advantage_mode="$ADVANTAGE_MODE"
  agent.config.rloo_advantage_margin=0.01
  agent.config.rloo_advantage_scale=0.20
  agent.config.rloo_advantage_clip=1.0
  agent.config.grpo_rollouts_per_mode="$ROLLOUTS_PER_MODE"
  agent.config.grpo_priority_manifest_path="$PRIORITY_CONFIG_PATH"
  agent.config.grpo_priority_sample_fraction="$PRIORITY_FRACTION"
  agent.config.generation_mode_weighting="$MODE_WEIGHTING"
  agent.config.generation_ddim_eta=1.0
  agent.config.generation_final_std=0.05
  agent.config.generation_trust_projection_mode="$GENERATION_TRUST_PROJECTION_MODE"
  agent.config.generation_trust_calibration_path="$GENERATION_TRUST_CALIBRATION_PATH"
  agent.config.selection_entropy_weight=0.0
  agent.config.kl_loss_weight=0.0
  agent.config.diffusion_truncation_timestep=8
  agent.config.diffusion_roll_timesteps=[8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  experiment_name="$EXPERIMENT_NAME"
)
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
