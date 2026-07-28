#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 audit|formal H|HC EXPERIMENT [CUDA_DEVICES=0,1,2,3,4,5,6,7]"
  exit 2
fi

PHASE="$1"
BRANCH="$2"
EXPERIMENT="$3"
CUDA_DEVICES="${4:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_ROOT/planning/script/run_training.py" ]]; then
  NAVSIM_ROOT="$ROOT_DIR/navsim"
fi
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
INPUTS="$ROOT_DIR/artifacts/grpo_stage29/pilot/inputs.json"
INPUTS_SHA=775e8ac63c8b398e91809e8e4d7222a724f29552fc397dd3f69a215a8fc4d0f7

[[ "$PHASE" == audit || "$PHASE" == formal ]] || {
  echo "Stage29 pilot phase must be audit or formal"; exit 2;
}
[[ "$BRANCH" == H || "$BRANCH" == HC ]] || {
  echo "Stage29 pilot branch must be H or HC"; exit 2;
}
[[ "$EXPERIMENT" =~ ^stage29_pilot_(H|HC)_(audit|formal)_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid Stage29 experiment name: $EXPERIMENT"; exit 2;
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage29 pilot requires eight distinct GPU indices"; exit 2;
}
IFS=',' read -r -a DEVICES <<< "$CUDA_DEVICES"
declare -A SEEN=()
for device in "${DEVICES[@]}"; do
  [[ -z "${SEEN[$device]:-}" ]] || { echo "duplicate GPU $device"; exit 2; }
  SEEN[$device]=1
done
for path in "$PYTHON_BIN" "$TRAIN_ENTRY" "$INPUTS"; do
  [[ -e "$path" ]] || { echo "missing Stage29 pilot input: $path"; exit 2; }
done
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE"; exit 2; }
[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || {
  echo "refusing to reuse Stage29 experiment: $EXP_ROOT/$EXPERIMENT"; exit 2;
}

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$INPUTS" "$INPUTS_SHA" "$BRANCH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

path, expected, branch = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if sha(path) != expected:
    raise RuntimeError("Stage29 pilot input-freeze SHA drifted")
p = json.loads(path.read_text(encoding="utf-8"))
if not p.get("passed") or p.get("stage") != 29 or branch not in p["branches"]:
    raise RuntimeError("Stage29 pilot frozen branch drifted")
for key in (
    "public_checkpoint", "manifest", "deployment_selector",
    "deployment_calibration",
):
    value = Path(p[key])
    if not value.is_file() or sha(value) != p[f"{key}_sha256"]:
        raise RuntimeError(f"Stage29 pilot locked input drifted: {key}")
calibration = json.loads(Path(p["deployment_calibration"]).read_text())
if not calibration.get("passed"):
    raise RuntimeError("Stage29 requires passing S-multi calibration")
entry = p["branches"][branch]
print(p["public_checkpoint"])
print(p["manifest"])
print(entry["training_mode"])
print(p["deployment_selector"])
print(p["deployment_calibration"])
print(calibration["residual_margin"])
print(calibration["risk_threshold"])
print(calibration["ood_threshold"])
print(str(entry["conditional_regularization"]).lower())
PY
)
[[ "${#LOCKED[@]}" -eq 9 ]] || { echo "failed to resolve Stage29 inputs"; exit 2; }
PUBLIC="${LOCKED[0]}"
MANIFEST="${LOCKED[1]}"
TRAINING_MODE="${LOCKED[2]}"
SELECTOR="${LOCKED[3]}"
CALIBRATION="${LOCKED[4]}"
MARGIN="${LOCKED[5]}"
RISK="${LOCKED[6]}"
OOD="${LOCKED[7]}"
CONDITIONAL="${LOCKED[8]}"

MAX_EPOCHS=4
LIMIT_BATCHES=1.0
CHECKPOINT_EVERY=0
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_BATCHES=8
  CHECKPOINT_EVERY=1
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
fi

TRAIN_ARGS=(
  agent=diffusiondrive_agent
  train_test_split=trainval
  cache_path="$TRAIN_CACHE"
  use_cache_without_dataset=true
  force_cache_computation=false
  seed=0
  dataloader.params.batch_size=1
  dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS"
  trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0
  trainer.params.accumulate_grad_batches=8
  trainer.params.gradient_clip_val=1.0
  trainer.params.accelerator=gpu
  trainer.params.strategy=ddp
  trainer.params.precision=32-true
  +trainer.params.devices=8
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PUBLIC"
  agent.reference_checkpoint_path="$PUBLIC"
  agent.lr=1e-6
  agent.config.grpo_training_mode="$TRAINING_MODE"
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
  agent.config.grpo_reward_mode=pdms
  agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy
  agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0
  agent.config.selection_rank_loss_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0
  agent.config.selector_generation_kl_weight=0.0
  agent.config.policy_loss_weight=0.0
  agent.config.kl_loss_weight=0.0
  agent.config.generation_policy_loss_weight=1.0
  agent.config.generation_kl_loss_weight=0.0
  agent.config.generation_adaptive_kl_enabled=false
  agent.config.diffgrpo_bc_weight=0.1
  agent.config.diffgrpo_step_discount=0.6
  agent.config.diffgrpo_logprob_reduction=mean
  agent.config.diffgrpo_group_size=8
  agent.config.diffgrpo_base_advantage_clip=2.0
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.stage29_headroom_low=0.75
  agent.config.stage29_headroom_high=0.90
  agent.config.stage29_delta_scale_floor=0.002
  agent.config.stage29_rank_weight=0.5
  agent.config.stage29_conditional_regularization="$CONDITIONAL"
  agent.config.stage29_fixed_bc_weight=0.1
  agent.config.stage29_fixed_kl_weight=0.1
  agent.config.stage29_hard_bc_weight=0.05
  agent.config.stage29_mature_bc_weight=0.1
  agent.config.stage29_hard_kl_weight=0.05
  agent.config.stage29_mature_kl_weight=0.5
  agent.config.stage29_safety_kl_weight=0.5
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
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT"
)

if [[ "${GRPO_STAGE29_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PASS Stage29 pilot preflight phase=$PHASE branch=$BRANCH"
  echo "Mode: $TRAINING_MODE"
  echo "Conditional regularization: $CONDITIONAL"
  echo "Manifest: $MANIFEST"
  exit 0
fi

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
