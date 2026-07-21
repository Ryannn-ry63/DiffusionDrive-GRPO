from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "scripts/training/run_diffusiondrive_grpo_stage14.sh"


def _run(*args):
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage14_runner_rejects_bad_arity_and_phase_before_gpu_access():
    missing = _run()
    assert missing.returncode == 2
    assert "Usage:" in missing.stdout

    bad_phase = _run("resume", "0", "stage14_bad", "0")
    assert bad_phase.returncode == 2
    assert "PHASE must be audit or formal" in bad_phase.stdout


def test_stage14_runner_rejects_seed_device_and_name():
    bad_seed = _run("audit", "3", "stage14_bad", "0")
    assert bad_seed.returncode == 2
    assert "SEED must be 0, 1, or 2" in bad_seed.stdout

    bad_device = _run("audit", "0", "stage14_bad", "8")
    assert bad_device.returncode == 2
    assert "CUDA_DEVICE must be an integer from 0 to 7" in bad_device.stdout

    bad_name = _run("audit", "0", "wrong_name", "0")
    assert bad_name.returncode == 2
    assert "EXPERIMENT_NAME must start with stage14_" in bad_name.stdout


def test_stage14_runner_locks_registered_recipe():
    text = RUNNER.read_text()
    required = (
        "dataloader.params.batch_size=2",
        "agent.lr=1e-6",
        "agent.config.grpo_training_mode=generation",
        "agent.config.grpo_decoder_gradient_scope=all_layers",
        "agent.config.grpo_old_policy_sync_steps=32",
        "agent.config.grpo_reward_mode=pdms",
        "agent.config.generation_policy_loss_weight=1.0",
        "agent.config.generation_kl_loss_weight=0.1",
        "agent.config.generation_advantage_mode=collision_truncated_intra_anchor",
        "agent.config.grpo_rollouts_per_mode=2",
        "agent.config.generation_mode_weighting=uniform",
        "agent.config.diffusion_roll_timesteps=[8,0]",
        "agent.config.diffusion_scheduler_num_inference_steps=125",
    )
    for value in required:
        assert value in text
    assert "GRPO_RESUME_CHECKPOINT" in text
    assert "MAX_STEPS=128" in text
    assert "CHECKPOINT_INTERVAL=64" in text
