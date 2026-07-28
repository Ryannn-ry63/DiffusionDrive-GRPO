#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 audit|formal DPEL|SCF HOLDOUT(0|1) EXPERIMENT [CUDA_DEVICES]"
  exit 2
fi
PHASE="$1"; BRANCH="$2"; HOLDOUT="$3"; EXPERIMENT="$4"
CUDA_DEVICES="${5:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
if [[ ! -f "$NAVSIM_ROOT/planning/script/run_training.py" ]]; then
  NAVSIM_ROOT="$ROOT_DIR/navsim"
fi
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
# Hydra's common config resolves the run directory and several dataset paths
# through environment variables.  The external scheduler does not guarantee
# that these are present, so export the validated paths here rather than
# relying on the caller's shell setup.
export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
PUBLIC=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS
SELECTOR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt
CALIBRATION="$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json"
CV_FREEZE="$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json"
PLAN="$ROOT_DIR/GRPO_STAGE32_SELECTOR_AWARE_FRONTIER_PLAN_20260726.md"
PLAN_SHA=5c5f3b148c8d03fe3c714558314ebd895a8f143b94fd10fb9bdf8c4ad61cefcd

[[ "$PHASE" == audit || "$PHASE" == formal ]] || { echo "bad phase"; exit 2; }
[[ "$BRANCH" == DPEL || "$BRANCH" == SCF ]] || { echo "bad branch"; exit 2; }
[[ "$HOLDOUT" =~ ^[01]$ ]] || { echo "holdout must be 0 or 1"; exit 2; }
[[ "$EXPERIMENT" =~ ^stage32_pilot_(DPEL|SCF)_fold[01]_(audit|formal)_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid Stage32 experiment: $EXPERIMENT"; exit 2;
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || { echo "Stage32 requires 8 GPU indices"; exit 2; }
for path in "$PYTHON_BIN" "$TRAIN_ENTRY" "$PUBLIC" "$SELECTOR" "$CALIBRATION" "$CV_FREEZE" "$PLAN"; do
  [[ -e "$path" ]] || { echo "missing Stage32 input: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA" ]] || {
  echo "Stage32 plan SHA drifted"; exit 2;
}
[[ -d "$TRAIN_CACHE" ]] || { echo "missing training cache: $TRAIN_CACHE"; exit 2; }
[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || { echo "refusing to reuse $EXPERIMENT"; exit 2; }

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$HOLDOUT" "$CALIBRATION" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
freeze=Path(sys.argv[1]); holdout=int(sys.argv[2]); calibration=Path(sys.argv[3])
p=json.loads(freeze.read_text()); entry=p['folds'][holdout]
if p.get('stage') != 30 or entry['holdout_fold'] != holdout:
    raise RuntimeError('Stage31 CV freeze drifted')
for path, expected in ((Path(entry['path']),entry['sha256']),
                       (Path(p['bucket_manifest']),p['bucket_manifest_sha256'])):
    if not path.is_file() or sha(path)!=expected:
        raise RuntimeError(f'drifted {path}')
c=json.loads(calibration.read_text())
if not c.get('passed'): raise RuntimeError('S-multi calibration did not pass')
print(entry['path']); print(p['bucket_manifest'])
print(c['residual_margin']); print(c['risk_threshold']); print(c['ood_threshold'])
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || { echo "failed Stage31 input resolution"; exit 2; }
TRAIN_MANIFEST="${LOCKED[0]}"; BUCKET_MANIFEST="${LOCKED[1]}"
MARGIN="${LOCKED[2]}"; RISK="${LOCKED[3]}"; OOD="${LOCKED[4]}"
if [[ "$BRANCH" == DPEL ]]; then
  TRAINING_MODE=stage32_public_deployed_extended
else
  TRAINING_MODE=stage32_selector_aware_frontier
fi

MAX_EPOCHS=12; LIMIT_BATCHES=1.0; CHECKPOINT_EVERY=48; EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1; LIMIT_BATCHES=8; CHECKPOINT_EVERY=1
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
fi
SEED=$((203200 + HOLDOUT))
TRAIN_ARGS=(
  agent=diffusiondrive_agent train_test_split=trainval
  cache_path="$TRAIN_CACHE" use_cache_without_dataset=true force_cache_computation=false
  seed="$SEED" dataloader.params.batch_size=1 dataloader.params.num_workers=4
  trainer.params.max_epochs="$MAX_EPOCHS" trainer.params.limit_train_batches="$LIMIT_BATCHES"
  trainer.params.limit_val_batches=0 trainer.params.accumulate_grad_batches=8
  trainer.params.gradient_clip_val=1.0 trainer.params.accelerator=gpu
  trainer.params.strategy=ddp trainer.params.precision=32-true
  +trainer.params.devices=8 +trainer.params.use_distributed_sampler=false
  +trainer.params.enable_progress_bar=false +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PUBLIC" agent.reference_checkpoint_path="$PUBLIC" agent.lr=1e-6
  agent.config.grpo_training_mode="$TRAINING_MODE"
  agent.config.generation_policy_algorithm=diffgrpo_deployed_selected_set
  agent.config.grpo_decoder_gradient_scope=all_layers agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$TRAIN_MANIFEST"
  agent.config.stage31_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage32_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage31_optimizer_steps_per_epoch=48 agent.config.stage31_gradient_accumulation=8
  agent.config.stage31_global_bucket_composition=[2,30,8,24]
  agent.config.stage31_plan_sha256=3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d
  agent.config.stage31_headroom_low=0.75 agent.config.stage31_headroom_high=0.90
  agent.config.stage31_delta_scale_floor=0.002 agent.config.stage31_rank_weight=0.5
  agent.config.stage31_bc_weight=0.1 agent.config.stage31_kl_weight=0.1
  agent.config.stage31_safety_kl_weight=0.5
  agent.config.stage32_plan_sha256="$PLAN_SHA"
  agent.config.stage32_scf_objective_revision=mean_normalized_bc_kl_v2
  agent.config.stage32_frontier_pool_size=4
  agent.config.stage32_frontier_risk_margin=0.10
  agent.config.stage32_frontier_owner_margin=0.001
  agent.config.stage32_frontier_weight=0.5
  agent.config.stage32_frontier_mature_cap=0.25
  agent.config.stage32_optimizer_steps_per_epoch=48
  agent.config.stage32_gradient_accumulation=8
  agent.config.stage32_global_bucket_composition=[2,30,8,24]
  agent.config.grpo_reward_mode=pdms agent.config.grpo_scene_weight_mode=uniform
  agent.config.selection_behavior_weighting=old_policy agent.config.selection_entropy_weight=0.0
  agent.config.selection_exploration_floor=0.0 agent.config.selection_rank_loss_weight=0.0
  agent.config.selector_consistency_kl_weight=0.0 agent.config.selector_generation_kl_weight=0.0
  agent.config.policy_loss_weight=0.0 agent.config.kl_loss_weight=0.0
  agent.config.generation_policy_loss_weight=1.0 agent.config.generation_kl_loss_weight=0.0
  agent.config.generation_adaptive_kl_enabled=false agent.config.diffgrpo_bc_weight=0.1
  agent.config.diffgrpo_step_discount=0.6 agent.config.diffgrpo_logprob_reduction=mean
  agent.config.diffgrpo_group_size=8 agent.config.diffgrpo_base_advantage_clip=2.0
  agent.config.diffgrpo_safety_regression_tolerance=0.000001
  agent.config.generation_advantage_mode=group_zscore agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32 agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0 agent.config.weight_decay=0.0
  agent.config.diffusion_truncation_timestep=32 agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT"
)

if [[ "${GRPO_STAGE32_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
    --cfg job --resolve "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" >/dev/null
  echo "PASS Stage32 pilot preflight phase=$PHASE branch=$BRANCH holdout=$HOLDOUT"
  echo "Mode: $TRAINING_MODE"
  echo "Train manifest: $TRAIN_MANIFEST"
  echo "Bucket manifest: $BUCKET_MANIFEST"
  exit 0
fi
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
