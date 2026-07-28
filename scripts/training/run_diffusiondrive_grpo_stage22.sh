#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "Usage: $0 audit|audit_ddp|development|retrain|formal BASE_CHECKPOINT EXPERIMENT_NAME EPOCHS [RESUME=none] [SEED=0] [CUDA_DEVICES=0] [MAX_STEPS=0]"
  exit 2
fi
PHASE="$1"; BASE_CHECKPOINT="$2"; EXPERIMENT_NAME="$3"; EPOCHS="$4"
RESUME="${5:-none}"; SEED="${6:-0}"; CUDA_DEVICES="${7:-0}"; MAX_STEPS="${8:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
MANIFEST_DIR="$ROOT_DIR/artifacts/grpo_stage22/manifests"
LOCKED_EPOCH_FILE="${STAGE22_LOCKED_EPOCH_FILE:-}"
EXPECTED_BASE_SHA="59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"

[[ "$PHASE" =~ ^(audit|audit_ddp|development|retrain|formal)$ ]] || { echo "invalid phase"; exit 2; }
[[ -f "$BASE_CHECKPOINT" && -d "$TRAIN_CACHE" ]] || { echo "missing base checkpoint/cache"; exit 2; }
[[ "$EXPERIMENT_NAME" =~ ^stage22_[A-Za-z0-9_.-]+$ ]] || { echo "experiment must start stage22_"; exit 2; }
[[ "$EPOCHS" =~ ^[1-4]$ && "$SEED" =~ ^[0-2]$ && "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "invalid numeric argument"; exit 2; }
[[ "$RESUME" == none || -f "$RESUME" ]] || { echo "resume checkpoint missing"; exit 2; }
ACTUAL_BASE_SHA="$("$PYTHON_BIN" -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(x) for x in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())' "$BASE_CHECKPOINT")"
[[ "$ACTUAL_BASE_SHA" == "$EXPECTED_BASE_SHA" ]] || { echo "official base SHA mismatch"; exit 2; }

DEVICES=1; BATCH_SIZE=1; ACCUMULATE=1; STRATEGY=auto
MANIFEST="$MANIFEST_DIR/smoke32_manifest.json"
CHECKPOINT_EVERY=0
if [[ "$PHASE" == audit ]]; then
  [[ "$CUDA_DEVICES" =~ ^[0-7]$ && ( "$MAX_STEPS" == 8 || "$MAX_STEPS" == 32 ) ]] || { echo "audit requires one GPU and 8/32 max steps"; exit 2; }
  CHECKPOINT_EVERY=8
elif [[ "$PHASE" == audit_ddp ]]; then
  [[ "$EPOCHS" == 2 && "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ && "$MAX_STEPS" == 64 ]] || { echo "audit_ddp requires 8 GPUs, 2 epoch ceiling, and 64 global updates"; exit 2; }
  DEVICES=8; ACCUMULATE=8; STRATEGY=ddp
  MANIFEST="$MANIFEST_DIR/fit_select_manifest.json"
elif [[ "$PHASE" == development ]]; then
  [[ "$EPOCHS" == 4 && "$MAX_STEPS" == 0 && "$RESUME" == none && "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "development requires fresh 8-GPU 4-epoch run"; exit 2; }
  DEVICES=8; ACCUMULATE=8; STRATEGY=ddp
  MANIFEST="$MANIFEST_DIR/fit_select_manifest.json"
else
  [[ "$MAX_STEPS" == 0 && "$RESUME" == none && "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "$PHASE requires fresh 8-GPU training"; exit 2; }
  [[ -n "$LOCKED_EPOCH_FILE" && -f "$LOCKED_EPOCH_FILE" ]] || { echo "$PHASE requires STAGE22_LOCKED_EPOCH_FILE"; exit 2; }
  LOCKED_EPOCH="$("$PYTHON_BIN" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_epoch"]))' "$LOCKED_EPOCH_FILE")"
  [[ "$EPOCHS" == "$LOCKED_EPOCH" ]] || { echo "epoch differs from locked Stage22 selection"; exit 2; }
  DEVICES=8; ACCUMULATE=8; STRATEGY=ddp
  if [[ "$PHASE" == retrain ]]; then
    MANIFEST="$MANIFEST_DIR/retrain_fit_manifest.json"
  else
    MANIFEST="$MANIFEST_DIR/formal_all_manifest.json"
  fi
fi
[[ -f "$MANIFEST" ]] || { echo "Stage22 manifest missing: $MANIFEST"; exit 2; }

EXTRA_ARGS=()
[[ "$MAX_STEPS" != 0 ]] && EXTRA_ARGS+=(+trainer.params.max_steps="$MAX_STEPS")
[[ "$RESUME" != none ]] && EXTRA_ARGS+=(+resume_checkpoint_path="$RESUME")
TRAIN_ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed="$SEED" dataloader.params.batch_size="$BATCH_SIZE" dataloader.params.num_workers=4
  trainer.params.max_epochs="$EPOCHS" trainer.params.limit_train_batches=1.0
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches="$ACCUMULATE"
  trainer.params.gradient_clip_val=1.0 trainer.params.gradient_clip_algorithm=norm
  trainer.params.accelerator=gpu trainer.params.strategy="$STRATEGY" trainer.params.precision=32-true
  +trainer.params.devices="$DEVICES" +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$BASE_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT"
  agent.lr=3e-5 agent.config.weight_decay=0.0
  agent.config.grpo_training_mode=diffgrpo_paired_residual
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.inference_selector_source=reference agent.config.grpo_reward_mode=pdms
  agent.config.grpo_scene_weight_mode=uniform agent.config.selection_behavior_weighting=old_policy
  agent.config.selection_entropy_weight=0.0 agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0 agent.config.selector_consistency_kl_weight=0.0
  agent.config.selector_generation_kl_weight=0.0 agent.config.policy_loss_weight=0.0
  agent.config.kl_loss_weight=0.0 agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.0 agent.config.generation_adaptive_kl_enabled=false
  agent.config.generation_policy_algorithm=diffgrpo_paired_residual
  agent.config.diffgrpo_base_scale=0.10 agent.config.diffgrpo_base_advantage_clip=2.0
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.diffgrpo_paired_positive_margin=0.01 agent.config.diffgrpo_paired_negative_margin=0.01
  agent.config.diffgrpo_paired_mature_negative_margin=0.002
  agent.config.diffgrpo_paired_mature_reward_threshold=0.75
  agent.config.diffgrpo_paired_mature_negative_multiplier=2.0
  agent.config.diffgrpo_paired_regular_kl_weight=0.1 agent.config.diffgrpo_paired_mature_kl_weight=0.5
  agent.config.diffgrpo_paired_bootstrap_advantage_weight=0.25
  agent.config.diffgrpo_lora_rank=8 agent.config.diffgrpo_lora_alpha=8.0
  agent.config.diffgrpo_step_discount=0.6 agent.config.diffgrpo_logprob_reduction=mean
  agent.config.diffgrpo_group_size=8 agent.config.diffgrpo_selected_mode_manifest_path="$MANIFEST"
  agent.config.diffgrpo_train_manifest_path= agent.config.generation_advantage_mode=group_zscore
  agent.config.grpo_rollouts_per_mode=1 agent.config.generation_mode_weighting=uniform
  agent.config.generation_trust_projection_mode=none agent.config.grpo_old_policy_sync_steps=32
  agent.config.grpo_priority_manifest_path= agent.config.grpo_priority_sample_fraction=0.0
  agent.config.diffusion_truncation_timestep=32 agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125 agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT_NAME"
)
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" navsim/planning/script/run_training.py "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
