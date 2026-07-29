#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 audit|formal|chain HOLDOUT(0|1) EXPERIMENT [CUDA_DEVICES]"
  exit 2
fi

PHASE="$1"
HOLDOUT="$2"
EXPERIMENT="$3"
CUDA_DEVICES="${4:-0,1,2,3,4,5,6,7}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsimH100/bin/python}"
NAVSIM_ROOT="${NAVSIM_DEVKIT_ROOT:-$ROOT_DIR/navsim}"
[[ -f "$NAVSIM_ROOT/planning/script/run_training.py" ]] || NAVSIM_ROOT="$ROOT_DIR/navsim"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp}"
TRAIN_CACHE="${GRPO_TRAIN_CACHE:-$EXP_ROOT/training_cache}"
TRAIN_ENTRY="$NAVSIM_ROOT/planning/script/run_training.py"
PUBLIC="${STAGE37_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE37_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE37_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE37_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
PLAN="$ROOT_DIR/GRPO_STAGE37_BISTATE_PROJECTED_JFI_PLAN_20260728.md"
PLAN_SHA="4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
SELF="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage37_generator.sh"
FREEZER="$ROOT_DIR/scripts/evaluation/freeze_grpo_stage37_checkpoints.py"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage37_checkpoint.py"

[[ "$PHASE" == audit || "$PHASE" == formal || "$PHASE" == chain ]] || {
  echo "phase must be audit, formal, or chain"; exit 2;
}
[[ "$HOLDOUT" =~ ^[01]$ ]] || { echo "holdout must be 0 or 1"; exit 2; }
[[ "$EXPERIMENT" =~ ^stage37_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid Stage37 experiment name: $EXPERIMENT"; exit 2;
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage37 requires exactly eight GPU indices"; exit 2;
}
for path in \
  "$PYTHON_BIN" "$TRAIN_ENTRY" "$PUBLIC" "$SELECTOR" "$CALIBRATION" \
  "$CV_FREEZE" "$PLAN" "$FREEZER" "$AUDITOR"; do
  [[ -e "$path" ]] || { echo "missing Stage37 input: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA" ]] || {
  echo "Stage37 plan SHA drifted"; exit 2;
}
[[ "$(sha256sum "$PUBLIC" | awk '{print $1}')" == \
  "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b" ]] || {
  echo "Stage37 public initializer SHA drifted"; exit 2;
}
[[ -d "$TRAIN_CACHE" ]] || {
  echo "missing Stage37 training cache: $TRAIN_CACHE"; exit 2;
}

export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

if [[ "$PHASE" == chain ]]; then
  "$SELF" audit "$HOLDOUT" "${EXPERIMENT}_audit" "$CUDA_DEVICES"
  "$SELF" formal "$HOLDOUT" "${EXPERIMENT}_formal" "$CUDA_DEVICES"
  echo "PASS Stage37 queue-safe generator chain holdout=$HOLDOUT"
  exit 0
fi

[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || {
  echo "refusing to reuse Stage37 experiment directory $EXP_ROOT/$EXPERIMENT"
  exit 2
}

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$HOLDOUT" "$CALIBRATION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

freeze = json.loads(Path(sys.argv[1]).read_text())
holdout = int(sys.argv[2])
calibration = json.loads(Path(sys.argv[3]).read_text())
entry = freeze["folds"][holdout]
bucket = Path(freeze["bucket_manifest"])
if freeze.get("stage") != 30 or entry["holdout_fold"] != holdout:
    raise RuntimeError("Stage37 CV freeze semantics drifted")
for path, expected in (
    (Path(entry["path"]), entry["sha256"]),
    (bucket, freeze["bucket_manifest_sha256"]),
):
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"Stage37 frozen manifest drifted: {path}")
if not calibration.get("passed"):
    raise RuntimeError("Stage37 frozen Stage25 calibration did not pass")
print(entry["path"])
print(bucket)
print(calibration["residual_margin"])
print(calibration["risk_threshold"])
print(calibration["ood_threshold"])
PY
)
[[ "${#LOCKED[@]}" -eq 5 ]] || {
  echo "failed Stage37 manifest/calibration resolution"; exit 2;
}
TRAIN_MANIFEST="${LOCKED[0]}"
BUCKET_MANIFEST="${LOCKED[1]}"
MARGIN="${LOCKED[2]}"
RISK="${LOCKED[3]}"
OOD="${LOCKED[4]}"

MAX_EPOCHS=4
LIMIT_BATCHES=1.0
CHECKPOINT_EVERY=48
EXTRA_ARGS=()
if [[ "$PHASE" == audit ]]; then
  MAX_EPOCHS=1
  LIMIT_BATCHES=8
  CHECKPOINT_EVERY=1
  EXTRA_ARGS+=(+trainer.params.max_steps=1)
fi
SEED=$((203700 + HOLDOUT))
TRAIN_ARGS=(
  agent=diffusiondrive_agent
  train_test_split=trainval
  split=trainval
  cache_path="$TRAIN_CACHE"
  use_cache_without_dataset=true
  force_cache_computation=false
  seed="$SEED"
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
  +trainer.params.use_distributed_sampler=false
  +trainer.params.enable_progress_bar=false
  +trainer.params.log_every_n_steps=1
  agent.checkpoint_path="$PUBLIC"
  agent.reference_checkpoint_path="$PUBLIC"
  agent.lr=1e-6
  agent.config.grpo_training_mode=stage37_bistate_projected_deployment_grpo
  agent.config.generation_policy_algorithm=diffgrpo_bistate_projected_deployment
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$TRAIN_MANIFEST"
  agent.config.stage37_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage37_plan_sha256="$PLAN_SHA"
  agent.config.stage37_objective_revision=bistate_projected_deployment_v1
  agent.config.stage37_active_pool_width=4
  agent.config.stage37_selector_micro_batch_size=16
  agent.config.stage37_frontier_width=5
  agent.config.stage37_tail_elite_count=2
  agent.config.stage37_frontier_regression_tolerance=0.0001
  agent.config.stage37_tail_margin=0.001
  agent.config.stage37_advantage_scale=0.002
  agent.config.stage37_advantage_clip=2.0
  agent.config.stage37_counterfactual_weight=0.25
  agent.config.stage37_tail_weight=0.25
  agent.config.stage37_frontier_weight=0.25
  agent.config.stage37_projection_recovery_coefficient=0.25
  agent.config.stage37_projection_epsilon=1e-12
  agent.config.stage37_bc_weight=0.1
  agent.config.stage37_kl_weight=0.1
  agent.config.stage37_safety_kl_weight=0.5
  agent.config.stage37_optimizer_steps_per_epoch=48
  agent.config.stage37_gradient_accumulation=8
  agent.config.stage37_global_bucket_composition=[2,30,8,24]
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
  agent.config.generation_advantage_mode=group_zscore
  agent.config.grpo_rollouts_per_mode=1
  agent.config.generation_mode_weighting=uniform
  agent.config.generation_trust_projection_mode=none
  agent.config.grpo_old_policy_sync_steps=32
  agent.config.grpo_priority_manifest_path=
  agent.config.grpo_priority_sample_fraction=0.0
  agent.config.weight_decay=0.0
  agent.config.diffusion_truncation_timestep=32
  agent.config.diffusion_roll_timesteps=[32,24,16,8,0]
  agent.config.diffusion_scheduler_num_inference_steps=125
  agent.config.grpo_checkpoint_save_top_k=-1
  agent.config.grpo_checkpoint_every_n_train_steps="$CHECKPOINT_EVERY"
  experiment_name="$EXPERIMENT"
)

cd "$ROOT_DIR"
if [[ "${GRPO_STAGE37_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
    --cfg job --resolve "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" >/dev/null
  echo "PASS Stage37 generator preflight phase=$PHASE holdout=$HOLDOUT"
  echo "Repository: $ROOT_DIR"
  echo "Training entry: $TRAIN_ENTRY"
  echo "Public initializer/reference: $PUBLIC"
  echo "Train manifest: $TRAIN_MANIFEST"
  echo "Bucket manifest: $BUCKET_MANIFEST"
  echo "Experiment root: $EXP_ROOT/$EXPERIMENT"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"

ARTIFACT_DIR="$ROOT_DIR/artifacts/grpo_stage37/generator/fold${HOLDOUT}/${PHASE}"
FREEZE="$ARTIFACT_DIR/checkpoints.json"
AUDIT="$ARTIFACT_DIR/audit.json"
[[ ! -e "$FREEZE" && ! -e "$AUDIT" ]] || {
  echo "refusing to overwrite Stage37 artifacts in $ARTIFACT_DIR"; exit 2;
}
"$PYTHON_BIN" "$FREEZER" \
  --phase "$PHASE" --holdout "$HOLDOUT" \
  --experiment-root "$EXP_ROOT/$EXPERIMENT" --output "$FREEZE"
"$PYTHON_BIN" "$AUDITOR" \
  --phase "$PHASE" --holdout "$HOLDOUT" \
  --base "$PUBLIC" --selector "$SELECTOR" --calibration "$CALIBRATION" \
  --cv-freeze "$CV_FREEZE" --plan "$PLAN" --freeze "$FREEZE" \
  --output "$AUDIT"

echo "PASS Stage37 generator phase=$PHASE holdout=$HOLDOUT"
echo "Run: $EXP_ROOT/$EXPERIMENT"
echo "Checkpoint freeze: $FREEZE"
echo "Audit: $AUDIT"
