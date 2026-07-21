import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "scripts/training/run_diffusiondrive_grpo_stage15.sh"
EVAL_RUNNER = ROOT / "scripts/evaluation/run_diffusiondrive_grpo_stage15_eval.sh"
MANIFESTS = ROOT / "artifacts/grpo_stage15/manifests"


def run(*args):
    return subprocess.run(
        ["bash", str(RUNNER), *args], cwd=ROOT, capture_output=True, text=True
    )


def test_stage15_runner_rejects_bad_inputs_before_gpu_access():
    assert run().returncode == 2
    assert run("resume", "stage15_bad", "0").returncode == 2
    assert run("audit", "wrong_name", "0").returncode == 2
    assert run("audit", "stage15_bad", "8").returncode == 2


def test_stage15_runner_locks_recipe():
    text = RUNNER.read_text()
    required = (
        "agent.lr=1e-4",
        "dataloader.params.batch_size=2",
        "trainer.params.precision=32-true",
        "agent.config.grpo_training_mode=value_selector",
        "agent.config.value_selector_num_heads=3",
        "agent.config.value_selector_top_k=2",
        "agent.config.value_selector_pair_reward_gap=0.01",
        "agent.config.value_selector_bootstrap_fraction=0.8",
        "agent.config.diffusion_roll_timesteps=[8,0]",
        "MAX_EPOCHS=4",
        "--expected-global-step 6426",
        "+resume_checkpoint_path=",
    )
    for item in required:
        assert item in text


def test_stage15_evaluator_runner_locks_value_top2_protocol():
    text = EVAL_RUNNER.read_text()
    required = (
        "--selector-logits-source value_top2",
        "--value-selector-checkpoint",
        "--value-selector-calibration-margin",
        "--truncation-timestep 8",
        "--roll-timesteps 8 0",
        "--scheduler-num-inference-steps 125",
        "--bootstrap-seed 20260721",
    )
    for item in required:
        assert item in text


def test_stage15_manifests_have_locked_counts_and_are_disjoint():
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
    assert not (token_sets[0] & token_sets[1])
    assert not (token_sets[0] & token_sets[2])
    assert not (token_sets[1] & token_sets[2])


def test_stage15_plan_and_interfaces_are_preregistered():
    plan = (ROOT / "GRPO_STAGE15_TRAJECTORY_VALUE_SELECTOR_PLAN.md").read_text()
    config = (ROOT / "navsim/agents/diffusiondrive/transfuser_config.py").read_text()
    evaluator = (ROOT / "scripts/evaluation/evaluate_grpo_schedule.py").read_text()
    assert "row 4 minus row 3" in plan
    assert 'value_selector_calibration_margin: float = -1.0' in config
    assert (
        'choices=("current", "reference", "value_top2", "paired_tail_risk")'
        in evaluator
    )
