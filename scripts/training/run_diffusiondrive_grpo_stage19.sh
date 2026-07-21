#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "Usage: $0 audit|development|formal INPUT_CHECKPOINT EXPERIMENT_NAME EPOCHS [RESUME=none] [SEED=0] [CUDA_DEVICES=0] [MAX_STEPS=0]"
  exit 2
fi

PHASE="$1"; INPUT_CHECKPOINT="$2"; EXPERIMENT_NAME="$3"; EPOCHS="$4"
RESUME="${5:-none}"; SEED="${6:-0}"; CUDA_DEVICES="${7:-0}"; MAX_STEPS="${8:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
MANIFEST_DIR="$ROOT_DIR/artifacts/grpo_stage19/manifests"
LOCKED_EPOCH_FILE="${STAGE19_LOCKED_EPOCH_FILE:-}"

[[ "$PHASE" == audit || "$PHASE" == development || "$PHASE" == formal ]] || { echo "invalid phase"; exit 2; }
[[ -f "$INPUT_CHECKPOINT" && -f "$BASE_CHECKPOINT" && -d "$TRAIN_CACHE" ]] || { echo "missing checkpoint/cache"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage19_[A-Za-z0-9_.-]+$ ]] || { echo "experiment must start stage19_"; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-2]$ && "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "invalid numeric argument"; exit 2; }
[[ "$RESUME" == none || -f "$RESUME" ]] || { echo "resume checkpoint missing"; exit 2; }

DEVICES=1; BATCH_SIZE=2; ACCUMULATE=1; STRATEGY=auto
MANIFEST="$MANIFEST_DIR/smoke64_manifest.json"
if [[ "$PHASE" == audit ]]; then
  [[ "$CUDA_DEVICES" =~ ^[0-7]$ && ( "$MAX_STEPS" == 8 || "$MAX_STEPS" == 32 ) ]] || { echo "audit requires one GPU and 8/32 steps"; exit 2; }
elif [[ "$PHASE" == development ]]; then
  [[ "$EPOCHS" == 10 && "$MAX_STEPS" == 0 && "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "development requires 8 GPUs, 10 epochs"; exit 2; }
  DEVICES=8; BATCH_SIZE=1; ACCUMULATE=8; STRATEGY=ddp
  MANIFEST="$MANIFEST_DIR/fit_manifest.json"
else
  [[ "$MAX_STEPS" == 0 && "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "formal requires 8 GPUs"; exit 2; }
  [[ -n "$LOCKED_EPOCH_FILE" && -f "$LOCKED_EPOCH_FILE" ]] || { echo "formal requires STAGE19_LOCKED_EPOCH_FILE"; exit 2; }
  LOCKED_EPOCH="$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_epoch"]))' "$LOCKED_EPOCH_FILE")"
  [[ "$EPOCHS" == "$LOCKED_EPOCH" ]] || { echo "formal epoch differs from locked selection"; exit 2; }
  DEVICES=8; BATCH_SIZE=1; ACCUMULATE=8; STRATEGY=ddp
  MANIFEST="$MANIFEST_DIR/all_manifest.json"
fi
[[ -f "$MANIFEST" ]] || { echo "Stage19 manifest missing: $MANIFEST"; exit 2; }

EXTRA_ARGS=()
[[ "$MAX_STEPS" != 0 ]] && EXTRA_ARGS+=(+trainer.params.max_steps="$MAX_STEPS")
[[ "$RESUME" != none ]] && EXTRA_ARGS+=(+resume_checkpoint_path="$RESUME")
TRAIN_ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed="$SEED" dataloader.params.batch_size="$BATCH_SIZE" dataloader.params.num_workers=4
  trainer.params.max_epochs="$EPOCHS" trainer.params.limit_train_batches=1.0
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches="$ACCUMULATE"
  trainer.params.accelerator=gpu trainer.params.strategy="$STRATEGY" trainer.params.precision=32-true
  +trainer.params.devices="$DEVICES" +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$INPUT_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT"
  agent.lr=1e-6 agent.config.grpo_training_mode=diffgrpo_selected_anchor
  agent.config.grpo_decoder_gradient_scope=all_layers agent.config.inference_selector_source=reference
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy
  agent.config.selection_entropy_weight=0.0 agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0 agent.config.selector_consistency_kl_weight=0.0
  agent.config.selector_generation_kl_weight=0.0 agent.config.policy_loss_weight=0.0
  agent.config.kl_loss_weight=0.0 agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.0 agent.config.generation_adaptive_kl_enabled=false
  agent.config.generation_policy_algorithm=diffgrpo_selected_anchor
  agent.config.diffgrpo_bc_weight=0.1 agent.config.diffgrpo_step_discount=0.6
  agent.config.diffgrpo_logprob_reduction=mean agent.config.diffgrpo_group_size=8
  agent.config.diffgrpo_selected_mode_manifest_path="$MANIFEST"
  agent.config.diffgrpo_train_manifest_path=
  agent.config.generation_advantage_mode=group_zscore agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32 agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0 agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1 experiment_name="$EXPERIMENT_NAME"
)

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" navsim/planning/script/run_training.py "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
