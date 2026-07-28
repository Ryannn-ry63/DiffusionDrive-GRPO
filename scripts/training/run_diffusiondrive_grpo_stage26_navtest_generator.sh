#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 audit|final EXPERIMENT [CUDA_DEVICES=0,1,2,3,4,5,6,7]"
  exit 2
fi
PHASE="$1"
EXPERIMENT="$2"
CUDA_DEVICES="${3:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/envs/navsimH100/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/envs/navsimH100/bin/python
  else
    PYTHON_BIN=/root/miniconda3/envs/navsim/bin/python
  fi
fi
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_ROOT/planning/script/run_training.py" ]]; then
  NAVSIM_ROOT="$ROOT_DIR/navsim"
fi
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
BASE="${GRPO_BASE_CHECKPOINT:-$EXP_ROOT/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
MANIFEST="$ROOT_DIR/artifacts/grpo_stage26/manifests/generator_train_full_6119.json"
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage26_stage25_selector_formal_seed26025/2026.07.23.13.16.13/lightning_logs/version_0/checkpoints/grpo-14-30570.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage26/stage26_selector_calibration_fold4.json"
FOLD4_GATE="$ROOT_DIR/artifacts/grpo_stage26/fold4/generator_gate.json"
FOLD5_REPORT="$ROOT_DIR/artifacts/grpo_stage26/fold5/final_report.json"
FINAL_AUDIT="$ROOT_DIR/artifacts/grpo_stage26/final_generator_audit.json"

[[ "$PHASE" == audit || "$PHASE" == final ]] || { echo "invalid phase"; exit 2; }
[[ "$EXPERIMENT" =~ ^stage26_navtest_generator_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid experiment"; exit 2;
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage26 NavTest generator requires exactly eight GPUs"; exit 2;
}
for path in "$BASE" "$MANIFEST" "$SELECTOR" "$CALIBRATION" "$FOLD4_GATE" \
  "$FOLD5_REPORT" "$FINAL_AUDIT" "$TRAIN_ENTRY"; do
  [[ -f "$path" ]] || { echo "missing locked Stage26 input: $path"; exit 2; }
done
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE"; exit 2; }

read -r MARGIN RISK OOD < <(
  "$PYTHON_BIN" -c '
import hashlib,json,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
base,manifest,selector,calibration,fold4,fold5,audit=sys.argv[1:]
assert sha(base) == "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
assert sha(manifest) == "1763ca12bfd1bf480a1ed6a47cc71f95ef31123517e500551dae24292620ef68"
assert sha(selector) == "77ee768c5f469292d81049192d48797f2f3fae4f80029ae6853dd9d9c6de4207"
assert sha(calibration) == "0c14b8f9531d64028bdf4d71f35b34418ab31ad4ae11ccc6b3c60a4325f53a0d"
assert sha(fold4) == "3436534cdca9e8aefd2b4410b25a1198be3c267bcd2dc7dd7c4efdd6fae6a5a2"
assert sha(fold5) == "80b951729b4d6041a0d7d02b14e18de1846fac013bdc98357acca4e3dc525e3f"
m=json.load(open(manifest)); s=m["summary"]
assert s["purpose"] == "navtest_generator_train_full_6119"
assert s["source_folds"] == [0,1,2,3,4,5]
assert s["count"] == 6119 and s["num_logs"] == 908
f4=json.load(open(fold4)); assert f4["passed"] and not f4["stop_before_fold5"]
f5=json.load(open(fold5)); assert f5["passed"] and not f5["stop_before_navtest"]
a=json.load(open(audit)); assert a["passed"] and a["fold5_authorized"]
c=json.load(open(calibration)); assert c["passed"]
print(c["residual_margin"],c["risk_threshold"],c["ood_threshold"])
' "$BASE" "$MANIFEST" "$SELECTOR" "$CALIBRATION" "$FOLD4_GATE" \
  "$FOLD5_REPORT" "$FINAL_AUDIT"
)

MAX_EPOCHS=2
LIMIT_BATCHES=1.0
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_BATCHES=8
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
fi
TRAIN_ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed=0 dataloader.params.batch_size=1 dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS"
  trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=8
  trainer.params.accelerator=gpu trainer.params.strategy=ddp
  trainer.params.precision=32-true +trainer.params.devices=8
  +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$BASE" agent.reference_checkpoint_path="$BASE"
  agent.lr=1e-6 agent.config.grpo_training_mode=diffgrpo_selected_set
  agent.config.generation_policy_algorithm=diffgrpo_selected_set
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$MANIFEST"
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy
  agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0
  agent.config.selector_generation_kl_weight=0.0
  agent.config.policy_loss_weight=0.0 agent.config.kl_loss_weight=0.0
  agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.0
  agent.config.generation_adaptive_kl_enabled=false
  agent.config.diffgrpo_bc_weight=0.1 agent.config.diffgrpo_step_discount=0.6
  agent.config.diffgrpo_logprob_reduction=mean
  agent.config.diffgrpo_group_size=8
  agent.config.diffgrpo_paired_regular_kl_weight=0.1
  agent.config.diffgrpo_paired_mature_kl_weight=0.5
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.generation_advantage_mode=group_zscore
  agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform
  agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32
  agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0
  agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1
  experiment_name="$EXPERIMENT"
)
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
