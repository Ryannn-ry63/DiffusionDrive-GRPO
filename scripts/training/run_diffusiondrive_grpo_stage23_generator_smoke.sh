#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 u8|u32|u64 EXPERIMENT GPU SELECTOR_CKPT CALIBRATION_JSON GATE_JSON"
  exit 2
fi
TIER="$1"; EXPERIMENT="$2"; GPU="$3"; SELECTOR="$4"; CALIBRATION="$5"; GATE="$6"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHECKPOINT="${GRPO_BASE_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAINING_CACHE="${NAVSIM_TRAINING_CACHE:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache}"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage23/manifests/selector_fit_manifest.json"

case "$TIER" in u8) ACCUMULATE=8 ;; u32) ACCUMULATE=32 ;; u64) ACCUMULATE=64 ;; *) echo "invalid tier"; exit 2 ;; esac
[[ "$EXPERIMENT" =~ ^stage23_generator_[A-Za-z0-9_.-]+$ && "$GPU" =~ ^[0-7]$ ]] || { echo "invalid experiment/GPU"; exit 2; }
for path in "$BASE_CHECKPOINT" "$SELECTOR" "$CALIBRATION" "$GATE" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "missing Stage23 input: $path"; exit 2; }
done
read -r MARGIN THRESHOLD < <(
  "$PYTHON_BIN" -c '
import json,sys
calibration=json.load(open(sys.argv[1])); gate=json.load(open(sys.argv[2]))
assert calibration["passed"] and gate["passed"] and not gate["stop_before_generator_training"]
assert calibration["selector_checkpoint_sha256"] == gate["selector_checkpoint_sha256"]
print(calibration["residual_margin"], calibration["selected"]["threshold"])
' "$CALIBRATION" "$GATE"
)

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent train_test_split=trainval \
  cache_path="$TRAINING_CACHE" use_cache_without_dataset=true force_cache_computation=false \
  seed=0 dataloader.params.batch_size=1 dataloader.params.num_workers=2 \
  trainer.params.max_epochs=1 trainer.params.limit_train_batches="$ACCUMULATE" \
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches="$ACCUMULATE" \
  +trainer.params.max_steps=1 trainer.params.accelerator=gpu trainer.params.strategy=auto \
  trainer.params.precision=32-true +trainer.params.devices=1 \
  +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1 \
  agent.checkpoint_path="$BASE_CHECKPOINT" agent.reference_checkpoint_path="$BASE_CHECKPOINT" \
  agent.lr=1e-6 agent.config.grpo_training_mode=diffgrpo_selected_set \
  agent.config.generation_policy_algorithm=diffgrpo_selected_set \
  agent.config.grpo_decoder_gradient_scope=all_layers \
  agent.config.grpo_decoder_layer0_lr_mult=0.1 \
  agent.config.inference_selector_source=trajectory_oof \
  agent.config.stage23_selector_checkpoint_path="$SELECTOR" \
  agent.config.stage23_selector_calibration_path="$CALIBRATION" \
  agent.config.stage23_selector_residual_margin="$MARGIN" \
  agent.config.stage23_selector_safety_threshold="$THRESHOLD" \
  agent.config.stage23_generator_train_manifest_path="$MANIFEST" \
  agent.config.policy_loss_weight=0.0 agent.config.kl_loss_weight=0.0 \
  agent.config.selector_generation_kl_weight=0.0 agent.config.generation_kl_loss_weight=0.0 \
  agent.config.diffgrpo_group_size=8 agent.config.diffgrpo_bc_weight=0.1 \
  agent.config.diffgrpo_step_discount=0.6 agent.config.diffgrpo_logprob_reduction=mean \
  agent.config.diffgrpo_paired_regular_kl_weight=0.1 \
  agent.config.diffgrpo_paired_mature_kl_weight=0.5 \
  agent.config.diffusion_truncation_timestep=32 \
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0] \
  agent.config.diffusion_scheduler_num_inference_steps=125 \
  agent.config.grpo_checkpoint_save_top_k=-1 experiment_name="$EXPERIMENT"
