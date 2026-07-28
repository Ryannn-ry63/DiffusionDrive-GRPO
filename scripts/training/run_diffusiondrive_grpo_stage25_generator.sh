#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 audit|development SELECTOR CALIBRATION GATE EXPERIMENT [CUDA_DEVICES=0,1,2,3,4,5,6,7]"
  exit 2
fi
PHASE="$1"; SELECTOR="$2"; CALIBRATION="$3"; GATE="$4"; EXPERIMENT="$5"
CUDA_DEVICES="${6:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/navsimH100/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/envs/navsimH100/bin/python
  else
    PYTHON_BIN=/root/miniconda3/envs/navsim/bin/python
  fi
fi
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-$EXP_ROOT/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"

[[ "$PHASE" == audit || "$PHASE" == development ]] || { echo "invalid phase"; exit 2; }
[[ "$EXPERIMENT" =~ ^stage25_generator_[A-Za-z0-9_.-]+$ ]] || { echo "invalid experiment"; exit 2; }
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "Stage25 formal route requires eight GPUs"; exit 2; }
for path in "$SELECTOR" "$CALIBRATION" "$GATE" "$BASE_CHECKPOINT" "$MANIFEST" "$TRAIN_ENTRY"; do
  [[ -f "$path" ]] || { echo "missing Stage25 generator input: $path"; exit 2; }
done
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE"; exit 2; }
read -r MARGIN RISK OOD < <(
  "$PYTHON_BIN" -c '
import hashlib,json,sys
calibration=json.load(open(sys.argv[1])); gate=json.load(open(sys.argv[2]))
selector_sha=hashlib.sha256(open(sys.argv[3],"rb").read()).hexdigest()
calibration_sha=hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()
assert calibration["stage"] == 25 and calibration["passed"]
assert gate["stage"] == 25 and gate["passed"] and not gate["stop_before_generator_training"]
assert calibration["selector_checkpoint_sha256"] == selector_sha == gate["selector_checkpoint_sha256"]
assert gate["calibration_sha256"] == calibration_sha
print(calibration["residual_margin"], calibration["risk_threshold"], calibration["ood_threshold"])
' "$CALIBRATION" "$GATE" "$SELECTOR"
)

MAX_EPOCHS=8; LIMIT_BATCHES=1.0; MAX_STEPS=0
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1; LIMIT_BATCHES=8; MAX_STEPS=1
fi
EXTRA_ARGS=()
[[ "$MAX_STEPS" != 0 ]] && EXTRA_ARGS+=(+trainer.params.max_steps="$MAX_STEPS")
TRAIN_ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed=0 dataloader.params.batch_size=1 dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=8
  trainer.params.accelerator=gpu trainer.params.strategy=ddp trainer.params.precision=32-true
  +trainer.params.devices=8 +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$BASE_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT"
  agent.lr=1e-6 agent.config.grpo_training_mode=diffgrpo_selected_set
  agent.config.generation_policy_algorithm=diffgrpo_selected_set
  agent.config.grpo_decoder_gradient_scope=all_layers agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$MANIFEST"
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy
  agent.config.selection_entropy_weight=0.0 agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0 agent.config.selector_consistency_kl_weight=0.0
  agent.config.selector_generation_kl_weight=0.0 agent.config.policy_loss_weight=0.0
  agent.config.kl_loss_weight=0.0 agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.0 agent.config.generation_adaptive_kl_enabled=false
  agent.config.diffgrpo_bc_weight=0.1 agent.config.diffgrpo_step_discount=0.6
  agent.config.diffgrpo_logprob_reduction=mean agent.config.diffgrpo_group_size=8
  agent.config.diffgrpo_paired_regular_kl_weight=0.1
  agent.config.diffgrpo_paired_mature_kl_weight=0.5
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.generation_advantage_mode=group_zscore agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32 agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0 agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1 experiment_name="$EXPERIMENT"
)
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
