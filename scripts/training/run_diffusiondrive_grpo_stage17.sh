#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "Usage: $0 audit|pilot|formal INPUT_CHECKPOINT EXPERIMENT_NAME EPOCHS [RESUME=none] [SEED=0] [CUDA_DEVICES=0] [MAX_STEPS=0]"
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
FIT_MANIFEST="$ROOT_DIR/artifacts/grpo_stage17/manifests/risk_fit_manifest.json"
VALIDATION_MANIFEST="$ROOT_DIR/artifacts/grpo_stage17/manifests/risk_validation_manifest.json"
SMOKE_MANIFEST="$ROOT_DIR/artifacts/grpo_stage17/manifests/risk_smoke64_manifest.json"
POSITIVE_WEIGHTS="${STAGE17_POSITIVE_WEIGHTS:-[1.0,1.0,1.0,1.0,1.0,1.0]}"

[[ "$PHASE" == audit || "$PHASE" == pilot || "$PHASE" == formal ]] || { echo "invalid phase"; exit 2; }
[[ -f "$INPUT_CHECKPOINT" && -f "$BASE_CHECKPOINT" ]] || { echo "locked checkpoint missing"; exit 2; }
[[ -d "$TRAINING_CACHE" && -f "$FIT_MANIFEST" && -f "$VALIDATION_MANIFEST" ]] || { echo "cache or manifest missing"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage17_[A-Za-z0-9_.-]+$ ]] || { echo "invalid experiment name"; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-2]$ && "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "invalid numeric argument"; exit 2; }
if [[ "$RESUME_CHECKPOINT" != none && ! -f "$RESUME_CHECKPOINT" ]]; then echo "resume checkpoint missing"; exit 2; fi

DEVICES=1
BATCH_SIZE=2
ACCUMULATE=1
STRATEGY=auto
TRAIN_MANIFEST="$FIT_MANIFEST"
if [[ "$PHASE" == audit ]]; then
  [[ "$CUDA_DEVICES" =~ ^[0-7]$ && ( "$MAX_STEPS" == 8 || "$MAX_STEPS" == 32 ) ]] || { echo "audit requires one GPU and MAX_STEPS 8 or 32"; exit 2; }
  TRAIN_MANIFEST="$SMOKE_MANIFEST"
elif [[ "$PHASE" == pilot ]]; then
  [[ "$EPOCHS" == 2 && "$MAX_STEPS" == 0 ]] || { echo "pilot is locked to 2 epochs"; exit 2; }
  if [[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]]; then
    DEVICES=8
    BATCH_SIZE=1
    ACCUMULATE=4
    STRATEGY=ddp
  else
    [[ "$CUDA_DEVICES" =~ ^[0-7]$ ]] || { echo "pilot requires one or eight GPUs"; exit 2; }
  fi
else
  [[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ && ( "$EPOCHS" == 12 || "$EPOCHS" == 24 ) && "$MAX_STEPS" == 0 ]] || { echo "formal requires eight GPUs and 12 or 24 epochs"; exit 2; }
  [[ -n "${STAGE17_POSITIVE_WEIGHTS:-}" ]] || { echo "formal requires STAGE17_POSITIVE_WEIGHTS"; exit 2; }
  DEVICES=8
  BATCH_SIZE=1
  ACCUMULATE=4
  STRATEGY=ddp
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
  agent.lr=1e-4 +agent.config.weight_decay=1e-4 \
  agent.config.grpo_training_mode=paired_tail_risk_selector \
  agent.config.inference_selector_source=reference \
  agent.config.generation_policy_algorithm=diffgrpo_full_chain \
  agent.config.paired_risk_train_manifest_path="$TRAIN_MANIFEST" \
  agent.config.paired_risk_validation_manifest_path="$VALIDATION_MANIFEST" \
  agent.config.paired_risk_positive_weights="$POSITIVE_WEIGHTS" \
  agent.config.paired_risk_selected_mode_weight=4.0 \
  agent.config.paired_risk_delta_loss_weight=0.25 \
  agent.config.diffusion_truncation_timestep=32 \
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0] \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  agent.config.grpo_checkpoint_save_top_k=-1 \
  experiment_name="$EXPERIMENT_NAME" "${EXTRA_ARGS[@]}"
