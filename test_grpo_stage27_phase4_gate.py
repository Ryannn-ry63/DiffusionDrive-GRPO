from scripts.evaluation.check_grpo_stage27_phase4_fold5 import (
    NOISES,
    summarize_epoch,
)


def _logs():
    return [f"log-{index % 151}" for index in range(2046)]


def test_stage27_phase4_epoch_gate_accepts_registered_positive_gain():
    values = [0.006] * 2046
    noise_values = {
        NOISES[0]: [0.006] * 1023,
        NOISES[1]: [0.006] * 1023,
    }
    components = {
        "collision": [0.0] * 2046,
        "ttc": [0.0] * 2046,
    }
    result = summarize_epoch(
        1, values, _logs(), noise_values, components
    )
    assert result["all_checks_pass"]
    assert result["whole_log_bootstrap_ci"][0] > 0
    assert result["pooled_mean_delta"] >= 0.005


def test_stage27_phase4_epoch_gate_rejects_safety_regression():
    values = [0.006] * 2046
    noise_values = {
        NOISES[0]: [0.006] * 1023,
        NOISES[1]: [0.006] * 1023,
    }
    components = {
        "collision": [-0.001] * 2046,
        "ttc": [0.0] * 2046,
    }
    result = summarize_epoch(
        2, values, _logs(), noise_values, components
    )
    assert not result["all_checks_pass"]
    assert not result["checks"][
        "collision_delta_at_least_negative_0.0005"
    ]
