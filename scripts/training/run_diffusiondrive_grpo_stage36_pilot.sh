#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 audit|formal|chain RGT HOLDOUT(0|1) EXPERIMENT [CUDA_DEVICES]"
  exit 2
fi

PHASE="$1"
BRANCH="$2"
HOLDOUT="$3"
EXPERIMENT="$4"
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
PUBLIC="${STAGE36_PUBLIC_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_navsim_88p1_PDMS}"
SELECTOR="${STAGE36_SELECTOR_CHECKPOINT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/stage27_selector_multi_stage25_formal_seed27125/2026.07.24.09.28.29/lightning_logs/version_0/checkpoints/grpo-17-36684.ckpt}"
CALIBRATION="${STAGE36_SELECTOR_CALIBRATION:-$ROOT_DIR/artifacts/grpo_stage27/selectors/multi/calibration.json}"
CV_FREEZE="${STAGE36_CV_FREEZE:-$ROOT_DIR/artifacts/grpo_stage30/manifests/cv/freeze.json}"
DPEL_FREEZE="$ROOT_DIR/artifacts/grpo_stage32/pilot/training/DPEL/fold${HOLDOUT}/formal/checkpoints.json"
PLAN="$ROOT_DIR/GRPO_STAGE36_REFERENCE_GATED_TAIL_NCD_PLAN_20260728.md"
PLAN_SHA="858e6688faa9dff6d9027254e6a0262c67e47582d51a5829cc12a9723a25e521"
SELF="$ROOT_DIR/scripts/training/run_diffusiondrive_grpo_stage36_pilot.sh"
FREEZER="$ROOT_DIR/scripts/evaluation/freeze_grpo_stage36_checkpoints.py"
AUDITOR="$ROOT_DIR/scripts/evaluation/audit_grpo_stage36_checkpoint.py"

[[ "$PHASE" == audit || "$PHASE" == formal || "$PHASE" == chain ]] || {
  echo "phase must be audit, formal, or chain"; exit 2;
}
[[ "$BRANCH" == RGT ]] || {
  echo "Stage36 mainline branch must be RGT"; exit 2;
}
[[ "$HOLDOUT" =~ ^[01]$ ]] || {
  echo "Stage36 pilot holdout must be 0 or 1"; exit 2;
}
[[ "$EXPERIMENT" =~ ^stage36_[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid Stage36 experiment name: $EXPERIMENT"; exit 2;
}
[[ "$CUDA_DEVICES" =~ ^[0-7](,[0-7]){7}$ ]] || {
  echo "Stage36 requires exactly eight GPU indices"; exit 2;
}
for path in \
  "$PYTHON_BIN" "$TRAIN_ENTRY" "$PUBLIC" "$SELECTOR" "$CALIBRATION" \
  "$CV_FREEZE" "$DPEL_FREEZE" "$PLAN" "$FREEZER" "$AUDITOR"; do
  [[ -e "$path" ]] || { echo "missing Stage36 input: $path"; exit 2; }
done
[[ "$(sha256sum "$PLAN" | awk '{print $1}')" == "$PLAN_SHA" ]] || {
  echo "Stage36 plan SHA drifted"; exit 2;
}
[[ -d "$TRAIN_CACHE" ]] || {
  echo "missing Stage36 training cache: $TRAIN_CACHE"; exit 2;
}

export NAVSIM_DEVKIT_ROOT="$NAVSIM_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/inspire/hdd/global_public/public_datas/NAVSIM/maps}"
export PYTHONPATH="$ROOT_DIR:$NAVSIM_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

# Queue-safe dependency: formal is submitted immediately but cannot start
# until this job's own one-step audit, checkpoint freeze, metric audit, and
# optimizer-boundary audit have all passed.
if [[ "$PHASE" == chain ]]; then
  AUDIT_EXPERIMENT="${EXPERIMENT}_audit"
  FORMAL_EXPERIMENT="${EXPERIMENT}_formal"
  if [[ "${GRPO_STAGE36_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    "$SELF" audit "$BRANCH" "$HOLDOUT" "$AUDIT_EXPERIMENT" "$CUDA_DEVICES"
    "$SELF" formal "$BRANCH" "$HOLDOUT" "$FORMAL_EXPERIMENT" "$CUDA_DEVICES"
    echo "PASS Stage36 queue-safe chain preflight holdout=$HOLDOUT"
    exit 0
  fi
  "$SELF" audit "$BRANCH" "$HOLDOUT" "$AUDIT_EXPERIMENT" "$CUDA_DEVICES"
  "$SELF" formal "$BRANCH" "$HOLDOUT" "$FORMAL_EXPERIMENT" "$CUDA_DEVICES"
  echo "PASS Stage36 queue-safe chain holdout=$HOLDOUT"
  exit 0
fi

[[ ! -e "$EXP_ROOT/$EXPERIMENT" ]] || {
  echo "refusing to reuse Stage36 experiment directory $EXP_ROOT/$EXPERIMENT"
  exit 2
}

readarray -t LOCKED < <(
  "$PYTHON_BIN" - "$CV_FREEZE" "$HOLDOUT" "$CALIBRATION" "$DPEL_FREEZE" <<'PY'
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
calibration_path = Path(sys.argv[3])
dpel_freeze_path = Path(sys.argv[4])
entry = freeze["folds"][holdout]
bucket = Path(freeze["bucket_manifest"])
if freeze.get("stage") != 30 or entry["holdout_fold"] != holdout:
    raise RuntimeError("Stage36 CV freeze semantics drifted")
for path, expected in (
    (Path(entry["path"]), entry["sha256"]),
    (bucket, freeze["bucket_manifest_sha256"]),
):
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"Stage36 frozen manifest drifted: {path}")
dpel = json.loads(dpel_freeze_path.read_text())
allowed = {
    0: "2c01b93fb35b246b6c897db149f38c13cc4a4259933f8ef21aee4d311dfee951",
    1: "b2bf4243e622235de4003938fba85240d2fd660ad0a7123b750cbc83abf6a557",
}
matches = [
    record for record in dpel.get("checkpoints", [])
    if int(record.get("global_step", -1)) == 192
]
if not (
    dpel.get("passed") is True
    and dpel.get("stage") == 32
    and dpel.get("branch") == "DPEL"
    and dpel.get("holdout_fold") == holdout
    and len(matches) == 1
    and matches[0].get("sha256") == allowed[holdout]
):
    raise RuntimeError("Stage36 fold-matched DPEL192 freeze drifted")
initializer = Path(matches[0]["path"])
if not initializer.is_file() or sha(initializer) != allowed[holdout]:
    raise RuntimeError(f"Stage36 DPEL192 initializer drifted: {initializer}")

calibration = json.loads(calibration_path.read_text())
if not calibration.get("passed"):
    raise RuntimeError("Stage36 frozen S-multi calibration did not pass")
print(entry["path"])
print(bucket)
print(initializer)
print(calibration["residual_margin"])
print(calibration["risk_threshold"])
print(calibration["ood_threshold"])
PY
)
[[ "${#LOCKED[@]}" -eq 6 ]] || {
  echo "failed Stage36 manifest/calibration/DPEL resolution"; exit 2;
}
TRAIN_MANIFEST="${LOCKED[0]}"
BUCKET_MANIFEST="${LOCKED[1]}"
INITIALIZER="${LOCKED[2]}"
MARGIN="${LOCKED[3]}"
RISK="${LOCKED[4]}"
OOD="${LOCKED[5]}"

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
SEED=$((203600 + HOLDOUT))
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
  agent.checkpoint_path="$INITIALIZER"
  agent.reference_checkpoint_path="$PUBLIC"
  agent.lr=1e-6
  agent.config.grpo_training_mode=stage36_reference_gated_tail_ncd_grpo
  agent.config.generation_policy_algorithm=diffgrpo_reference_gated_tail_ncd
  agent.config.grpo_decoder_gradient_scope=all_layers
  agent.config.grpo_decoder_layer0_lr_mult=0.1
  agent.config.inference_selector_source=trajectory_relative_harm_v3
  agent.config.stage25_selector_checkpoint_path="$SELECTOR"
  agent.config.stage25_selector_calibration_path="$CALIBRATION"
  agent.config.stage24_selector_residual_margin="$MARGIN"
  agent.config.stage25_selector_risk_threshold="$RISK"
  agent.config.stage24_selector_ood_threshold="$OOD"
  agent.config.stage23_generator_train_manifest_path="$TRAIN_MANIFEST"
  agent.config.stage36_bucket_manifest_path="$BUCKET_MANIFEST"
  agent.config.stage36_plan_sha256="$PLAN_SHA"
  agent.config.stage36_objective_revision=reference_gated_tail_ncd_v1
  agent.config.stage36_active_pool_width=4
  agent.config.stage36_selector_micro_batch_size=16
  agent.config.stage36_frontier_risk_margin=0.1
  agent.config.stage36_frontier_width=5
  agent.config.stage36_tail_elite_count=2
  agent.config.stage36_retention_tolerance=0.0001
  agent.config.stage36_tail_margin=0.001
  agent.config.stage36_tail_scale=0.002
  agent.config.stage36_tail_weight=0.25
  agent.config.stage36_retention_weight=0.25
  agent.config.stage36_counterfactual_weight=0.25
  agent.config.stage36_mature_positive_multiplier=0.25
  agent.config.stage36_advantage_eps=0.001
  agent.config.stage36_delta_scale_floor=0.002
  agent.config.stage36_headroom_low=0.75
  agent.config.stage36_headroom_high=0.9
  agent.config.stage36_rank_weight=0.5
  agent.config.stage36_advantage_clip=2.0
  agent.config.stage36_bc_weight=0.1
  agent.config.stage36_kl_weight=0.1
  agent.config.stage36_safety_kl_weight=0.5
  agent.config.stage36_optimizer_steps_per_epoch=48
  agent.config.stage36_gradient_accumulation=8
  agent.config.stage36_global_bucket_composition=[2,30,8,24]
  agent.config.stage36_signal_retention_fraction_min=0.01
  agent.config.stage36_signal_tail_fraction_min=0.005
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
if [[ "${GRPO_STAGE36_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
    --cfg job --resolve "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}" >/dev/null
  echo "PASS Stage36 preflight phase=$PHASE holdout=$HOLDOUT"
  echo "Repository: $ROOT_DIR"
  echo "Training entry: $TRAIN_ENTRY"
  echo "Train manifest: $TRAIN_MANIFEST"
  echo "DPEL192 initializer: $INITIALIZER"
  echo "Bucket manifest: $BUCKET_MANIFEST"
  echo "Experiment root: $EXP_ROOT/$EXPERIMENT"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "$TRAIN_ENTRY" \
  "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"

ARTIFACT_DIR="$ROOT_DIR/artifacts/grpo_stage36/pilot/training/RGT/fold${HOLDOUT}/${PHASE}"
FREEZE="$ARTIFACT_DIR/checkpoints.json"
AUDIT="$ARTIFACT_DIR/audit.json"
[[ ! -e "$FREEZE" && ! -e "$AUDIT" ]] || {
  echo "refusing to overwrite Stage36 audit artifacts in $ARTIFACT_DIR"; exit 2;
}
"$PYTHON_BIN" "$FREEZER" \
  --phase "$PHASE" \
  --holdout "$HOLDOUT" \
  --experiment-root "$EXP_ROOT/$EXPERIMENT" \
  --output "$FREEZE"
"$PYTHON_BIN" "$AUDITOR" \
  --phase "$PHASE" \
  --holdout "$HOLDOUT" \
  --base "$PUBLIC" \
  --selector "$SELECTOR" \
  --calibration "$CALIBRATION" \
  --cv-freeze "$CV_FREEZE" \
  --initializer "$INITIALIZER" \
  --plan "$PLAN" \
  --freeze "$FREEZE" \
  --output "$AUDIT"

echo "PASS Stage36 training phase=$PHASE branch=$BRANCH holdout=$HOLDOUT"
echo "Run: $EXP_ROOT/$EXPERIMENT"
echo "Checkpoint freeze: $FREEZE"
echo "Audit: $AUDIT"
