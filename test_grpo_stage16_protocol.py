"""Static and artifact protocol tests for Stage-16 execution."""

import json
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from navsim.agents.diffusiondrive.transfuser_agent import (
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "scripts/training/run_diffusiondrive_grpo_stage16.sh"
EVAL_RUNNER = ROOT / "scripts/evaluation/run_diffusiondrive_grpo_stage16_eval.sh"
GATE = ROOT / "scripts/evaluation/check_grpo_stage16_gate.py"
METRIC_GATE = ROOT / "scripts/training/check_grpo_stage16_metrics.py"
DEVELOPMENT_SELECTOR = ROOT / "scripts/evaluation/select_grpo_stage16_development.py"
MANIFESTS = ROOT / "artifacts/grpo_stage15/manifests"


def formal_stage16_config():
    return replace(
        TransfuserConfig(),
        grpo_training_mode="diffgrpo_full_chain",
        grpo_decoder_gradient_scope="all_layers",
        grpo_reward_mode="pdms",
        grpo_scene_weight_mode="uniform",
        selection_behavior_weighting="old_policy",
        generation_advantage_mode="group_zscore",
        generation_mode_weighting="uniform",
        generation_trust_projection_mode="none",
        grpo_rollouts_per_mode=1,
        grpo_old_policy_sync_steps=32,
        grpo_priority_manifest_path="",
        grpo_priority_sample_fraction=0.0,
        selection_entropy_weight=0.0,
        selection_exploration_floor=0.0,
        selection_rank_loss_weight=0.0,
        selector_consistency_kl_weight=0.0,
        grpo_clip_ratio=0.2,
        policy_loss_weight=0.0,
        kl_loss_weight=0.0,
        selector_generation_kl_weight=0.0,
        generation_policy_loss_weight=1.0,
        generation_kl_loss_weight=0.0,
        generation_adaptive_kl_enabled=False,
        generation_policy_algorithm="diffgrpo_full_chain",
        diffgrpo_bc_weight=0.1,
        diffgrpo_step_discount=0.6,
        diffgrpo_logprob_reduction="mean",
        inference_selector_source="reference",
        diffusion_truncation_timestep=32,
        diffusion_roll_timesteps=(32, 24, 16, 8, 0),
        diffusion_scheduler_num_inference_steps=125,
    )


def test_stage16_formal_config_accepts_only_locked_route():
    config = formal_stage16_config()
    validate_formal_grpo_config(config)
    with pytest.raises(ValueError, match="requires diffgrpo_bc_weight"):
        validate_formal_grpo_config(replace(config, diffgrpo_bc_weight=0.2))
    with pytest.raises(ValueError, match="requires inference_selector_source"):
        validate_formal_grpo_config(replace(config, inference_selector_source="current"))


def test_stage16_runner_rejects_bad_inputs_before_gpu_access():
    result = subprocess.run(
        ["bash", str(RUNNER)], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 2
    result = subprocess.run(
        ["bash", str(RUNNER), "audit", "/missing", "stage16_bad", "1", "none", "0", "0", "8"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_stage16_runners_lock_training_and_attribution_recipe():
    train = RUNNER.read_text()
    required_train = (
        "agent.lr=1e-6",
        "trainer.params.precision=32-true",
        "agent.config.grpo_training_mode=diffgrpo_full_chain",
        "agent.config.generation_policy_algorithm=diffgrpo_full_chain",
        "agent.config.diffgrpo_bc_weight=0.1",
        "agent.config.diffgrpo_step_discount=0.6",
        "agent.config.diffusion_roll_timesteps=[32,24,16,8,0]",
        "BATCH_SIZE=1",
        "ACCUMULATE=8",
        "development is locked to 10 epochs",
        "+resume_checkpoint_path=",
    )
    for item in required_train:
        assert item in train

    evaluate = EVAL_RUNNER.read_text()
    for item in (
        "--selector-logits-source reference",
        "TIMESTEPS=(8 0)",
        "TIMESTEPS=(32 24 16 8 0)",
        "ALGORITHM=diffgrpo_full_chain",
        "--generation-policy-algorithm",
        "--bootstrap-seed 20260721",
    ):
        assert item in evaluate


def test_stage16_inherits_the_locked_disjoint_train_calibration_test_manifests():
    expected = {
        "selector_train": (4283, "158db13ac32b3796df74c97fda67201d8d170a36b3ca4be61dcacfffe1a51d92"),
        "selector_calibration": (918, "88684e137a1e6aeaec6afec07f1cb85590777ae471de498b02db80ebb8dfe3de"),
        "selector_test": (918, "49d84f7a06f53cd7ee65ccaf776f521805e5823582569e93fd0bd4548f82f06c"),
    }
    token_sets = []
    for name, (count, digest) in expected.items():
        payload = json.loads((MANIFESTS / f"{name}_manifest.json").read_text())
        assert payload["summary"]["count"] == count
        assert payload["summary"]["ordered_token_sha256"] == digest
        tokens = {record["token"] for record in payload["records"]}
        assert len(tokens) == count
        token_sets.append(tokens)
    assert not token_sets[0].intersection(token_sets[1], token_sets[2])
    assert not token_sets[1].intersection(token_sets[2])


def _artifact(path, schedule, algorithm, reward):
    sha = "registered-base-sha"
    records = [
        {
            "token": f"token-{index:04d}",
            "selected_reward": reward,
            "selected_components": {
                name: 1.0
                for name in ("collision", "drivable", "progress", "ttc", "comfort", "direction")
            },
        }
        for index in range(918)
    ]
    payload = {
        "summary": {
            "selector_logits_source": "reference",
            "generation_policy_algorithm": algorithm,
            "checkpoint_sha256": sha,
            "reference_checkpoint_sha256": sha,
            "num_tokens": 918,
            "num_failures": 0,
            "completed": True,
            "schedule": {"roll_timesteps": schedule},
        },
        "records": records,
    }
    path.write_text(json.dumps(payload))


def test_stage16_schedule_gate_enforces_preregistered_tolerance(tmp_path):
    original = tmp_path / "original.json"
    full = tmp_path / "full.json"
    output = tmp_path / "gate.json"
    _artifact(original, [8, 0], "legacy_ppo", 0.5)
    _artifact(full, [32, 24, 16, 8, 0], "diffgrpo_full_chain", 0.496)
    command = [
        "python", str(GATE), "--gate", "schedule", "--base-original",
        str(original), "--base-full", str(full), "--output", str(output),
    ]
    passed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert json.loads(output.read_text())["passed"] is True

    _artifact(full, [32, 24, 16, 8, 0], "diffgrpo_full_chain", 0.49)
    failed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert failed.returncode == 1
    assert "full schedule selected delta is < -0.005" in failed.stdout


def test_stage16_plan_was_preregistered_before_execution():
    plan = (ROOT / "GRPO_STAGE16_FULL_CHAIN_DIFFGRPO_PLAN.md").read_text()
    assert "cell 3 minus cell 2 is the GRPO" in plan
    assert "gamma=0.6" in plan
    assert "No Stage-16 code or experiment" in plan


def test_stage16_metric_gate_locks_finite_active_and_frozen_diagnostics():
    text = METRIC_GATE.read_text()
    for item in (
        "train/diffgrpo_bc_loss_step",
        "train/diffgrpo_mean_current_log_prob_step",
        "train/diff_decoder_grad_norm_step",
        "train/perception_grad_norm_step",
        "train/classification_grad_norm_step",
        "0.6 ** 4",
    ):
        assert item in text


def test_stage16_development_selector_locks_epochs_safety_and_tie_break():
    text = DEVELOPMENT_SELECTOR.read_text()
    for item in (
        "EPOCHS = (1, 2, 4, 6, 8, 10)",
        'SAFETY = ("collision", "drivable", "ttc")',
        "best_reward - row[\"selected_reward\"] <= 0.0005",
        "key=lambda row: row[\"epoch\"]",
    ):
        assert item in text
