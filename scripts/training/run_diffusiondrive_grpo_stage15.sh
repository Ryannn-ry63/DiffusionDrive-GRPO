#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 audit|formal EXPERIMENT_NAME CUDA_DEVICE"
  echo "       $0 resume EXPERIMENT_NAME CUDA_DEVICE EPOCH3_CHECKPOINT"
  exit 2
fi

PHASE="$1"
EXPERIMENT_NAME="$2"
CUDA_DEVICE="$3"
RESUME_CHECKPOINT="${4:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/root/miniconda3/envs/navsim/bin/python"
BASE_CHECKPOINT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model"
GRPO_CHECKPOINT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_d_seed0_skl0.0_gkl0.1_eps0.001_dense0.1_mweightuniform_bs2/2026.07.16.14.53.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt"
TRAINING_CACHE="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache"
TRAIN_MANIFEST="$ROOT_DIR/artifacts/grpo_stage15/manifests/selector_train_manifest.json"

[[ "$PHASE" == audit || "$PHASE" == formal || "$PHASE" == resume ]] || { echo "invalid phase"; exit 2; }
if [[ "$PHASE" == resume ]]; then
  [[ $# -eq 4 && -f "$RESUME_CHECKPOINT" ]] || { echo "resume requires an epoch-3 checkpoint"; exit 2; }
else
  [[ $# -eq 3 ]] || { echo "$PHASE does not accept a resume checkpoint"; exit 2; }
fi
[[ "$EXPERIMENT_NAME" =~ ^stage15_[A-Za-z0-9_.-]+$ ]] || { echo "invalid experiment name"; exit 2; }
[[ "$CUDA_DEVICE" =~ ^[0-7]$ ]] || { echo "invalid CUDA device"; exit 2; }
[[ -f "$BASE_CHECKPOINT" && -f "$GRPO_CHECKPOINT" ]] || { echo "locked checkpoint missing"; exit 2; }
[[ -d "$TRAINING_CACHE" && -f "$TRAIN_MANIFEST" ]] || { echo "cache or manifest missing"; exit 2; }

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_DEVICE")"
[[ "$GPU_NAME" == *"RTX 4090"* ]] || { echo "Stage15 local gate requires RTX 4090; got $GPU_NAME"; exit 2; }

MAX_EPOCHS=4
LIMIT_TRAIN=1.0
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_TRAIN=8
  EXTRA_ARGS+=(+trainer.params.max_steps=8)
fi
if [[ "$PHASE" == resume ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/training/check_grpo_stage15_checkpoint.py" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --generator-checkpoint "$GRPO_CHECKPOINT" \
    --checkpoint "$RESUME_CHECKPOINT" \
    --expected-global-step 6426
  EXTRA_ARGS+=(+resume_checkpoint_path="$RESUME_CHECKPOINT")
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
  agent.checkpoint_path="$GRPO_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT" \
  agent.lr=1e-4 \
  agent.config.grpo_training_mode=value_selector \
  agent.config.value_selector_train_manifest_path="$TRAIN_MANIFEST" \
  agent.config.value_selector_num_heads=3 agent.config.value_selector_top_k=2 \
  agent.config.value_selector_pair_reward_gap=0.01 \
  agent.config.value_selector_bootstrap_fraction=0.8 \
  agent.config.grpo_checkpoint_save_top_k=-1 \
  agent.config.diffusion_truncation_timestep=8 \
  agent.config.diffusion_roll_timesteps=[8,0] \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  experiment_name="$EXPERIMENT_NAME" "${EXTRA_ARGS[@]}"
