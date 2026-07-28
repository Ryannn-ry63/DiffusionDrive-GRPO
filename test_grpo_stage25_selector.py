from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pytest

from scripts.evaluation.calibrate_grpo_stage25_selector import (
    incremental_mode_grid,
)

from navsim.agents.diffusiondrive.stage25_relative_harm_selector import (
    Stage25RelativeHarmSelector,
    compute_stage25_selector_loss,
    select_stage25_trajectory,
)
from navsim.agents.diffusiondrive.transfuser_agent import (
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


def _config():
    return SimpleNamespace(
        stage24_selector_num_members=8,
        stage24_selector_dim=32,
    )


def test_stage25_heads_are_independent_and_permutation_equivariant():
    torch.manual_seed(25025)
    selector = Stage25RelativeHarmSelector(_config()).eval()
    fingerprints = [
        float(torch.cat([
            parameter.detach().flatten() for parameter in member.parameters()
        ]).sum())
        for member in selector.members
    ]
    assert len(set(fingerprints)) == 8
    embeddings = torch.randn(2, 20, 8, 32)
    output = selector(embeddings)
    permutation = torch.randperm(20)
    moved = selector(embeddings[:, permutation])
    assert torch.allclose(
        output["harm_probabilities"][:, permutation],
        moved["harm_probabilities"],
    )


def test_stage25_relative_harm_boundary_and_loss_backpropagate():
    harm_logits = torch.randn(1, 20, 8, 3, requires_grad=True)
    catastrophe_logits = torch.randn(1, 20, 8, requires_grad=True)
    delta = torch.randn(1, 20, 8)
    components = torch.ones(1, 20, 6)
    # Equality is not harm; a strictly larger drop is harm.
    components[:, 1, 0] = 1.0 - 0.0005
    components[:, 2, 0] = 1.0 - 0.0006
    rewards = torch.full((1, 20), 0.8)
    rewards[:, 3] = 0.3
    result = compute_stage25_selector_loss({
        "stage25_harm_logits": harm_logits,
        "stage25_catastrophe_logits": catastrophe_logits,
        "stage24_delta_predictions": delta,
        "component_scores": components,
        "raw_rewards": rewards,
        "reward_valid_mask": torch.ones(1, 20, dtype=torch.bool),
        "stage24_fallback_mode": torch.tensor([0]),
        "stage24_member_training_mask": torch.ones(1, 8, dtype=torch.bool),
        "stage25_positive_weights": torch.tensor([10.0, 10.0, 10.0, 10.0]),
    })
    assert torch.isfinite(result["loss"])
    assert torch.isclose(
        result["stage25_harm_rate"], torch.tensor(1.0 / 60.0)
    )
    assert torch.isclose(
        result["stage25_catastrophe_rate"], torch.tensor(2.0 / 20.0)
    )
    result["loss"].backward()
    assert harm_logits.grad is not None and harm_logits.grad.abs().sum() > 0
    assert (
        catastrophe_logits.grad is not None
        and catastrophe_logits.grad.abs().sum() > 0
    )


def test_stage25_selection_uses_relative_risk_value_and_ood():
    reference = torch.arange(20, dtype=torch.float32).flip(0).unsqueeze(0)
    harm = torch.full((1, 20, 8, 3), 0.01)
    catastrophe = torch.full((1, 20, 8), 0.01)
    value = torch.zeros(1, 20, 8)
    value[:, 5] = 0.08
    value[:, 7] = 0.12
    value[:, 9] = 0.20
    catastrophe[:, 9] = 0.9
    embeddings = torch.zeros(1, 20, 4)
    embeddings[:, 7] = 10.0
    selected, diagnostics = select_stage25_trajectory(
        reference, harm, catastrophe, value, embeddings,
        residual_margin=0.01, risk_threshold=0.1,
        ood_mean=torch.zeros(4), ood_variance=torch.ones(4),
        ood_threshold=2.0,
    )
    assert selected.item() == 5
    assert diagnostics["ood_distance"][0, 7] > 2.0
    assert diagnostics["risk_ucb"][0, 9] > 0.1


def test_stage25_inference_has_no_internal_pdm_or_base_fallback():
    source = Path(
        "navsim/agents/diffusiondrive/stage25_relative_harm_selector.py"
    ).read_text()
    model_source = Path(
        "navsim/agents/diffusiondrive/transfuser_model_v2.py"
    ).read_text()
    assert "_compute_rewards_from_lazy_cache" not in source
    assert "ref_policy" not in source
    assert '"trajectory_relative_harm_v3",' in model_source


def test_stage25_incremental_threshold_sweep_exactly_matches_brute_force():
    # Includes exact threshold boundaries, tied value scores, fallback modes,
    # ineligible modes, and a risk above one that clipping must not admit.
    eligible = np.asarray([
        [False, True, True, True, True],
        [True, False, True, True, True],
        [True, True, False, False, True],
    ])
    risk = np.asarray([
        [0.0, 0.1, 0.2, 0.2, 1.2],
        [0.3, 0.0, -0.1, 0.8, 1.0],
        [0.4, 0.4, 0.0, 0.5, 0.9],
    ])
    score = np.asarray([
        [0.0, 0.2, 0.5, 0.5, 9.0],
        [0.4, 0.0, 0.1, 0.6, 0.7],
        [0.3, 0.3, 0.0, 4.0, 0.2],
    ])
    fallback = np.asarray([0, 1, 2])
    thresholds = np.asarray([0.0, 0.1, 0.199, 0.2, 0.4, 0.8, 0.9, 1.0])
    incremental = incremental_mode_grid(
        eligible, risk, score, fallback, thresholds
    )
    brute = []
    for threshold in thresholds:
        chosen = []
        for scene_index in range(eligible.shape[0]):
            admitted = eligible[scene_index] & (risk[scene_index] <= threshold)
            if admitted.any():
                chosen.append(int(np.argmax(np.where(
                    admitted, score[scene_index], -np.inf
                ))))
            else:
                chosen.append(int(fallback[scene_index]))
        brute.append(chosen)
    np.testing.assert_array_equal(incremental, np.asarray(brute))


def test_stage25_selected_set_formal_config_requires_frozen_inputs():
    config = TransfuserConfig()
    config.grpo_training_mode = "diffgrpo_selected_set"
    config.generation_policy_algorithm = "diffgrpo_selected_set"
    config.grpo_decoder_gradient_scope = "all_layers"
    config.grpo_decoder_layer0_lr_mult = 0.1
    config.inference_selector_source = "trajectory_relative_harm_v3"
    config.policy_loss_weight = 0.0
    config.kl_loss_weight = 0.0
    config.selector_generation_kl_weight = 0.0
    config.generation_kl_loss_weight = 0.0
    config.diffusion_truncation_timestep = 32
    config.diffusion_roll_timesteps = (32, 24, 16, 8, 0)
    with pytest.raises(ValueError, match="checkpoint and calibration"):
        validate_formal_grpo_config(config)
    config.stage25_selector_checkpoint_path = "selector.ckpt"
    config.stage25_selector_calibration_path = "calibration.json"
    validate_formal_grpo_config(config)
