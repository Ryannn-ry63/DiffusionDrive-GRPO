from types import SimpleNamespace

import torch

from navsim.agents.diffusiondrive.stage37_bistate_projected import (
    STAGE37_PLAN_SHA256,
    project_reward_gradient,
)
from navsim.agents.diffusiondrive.stage37_contract import (
    STAGE37_ALLOWED_INITIAL_SHA256,
    STAGE37_ROUTE_EXPECTED,
)
from navsim.agents.diffusiondrive.stage37_joint_feasible_selector import (
    Stage37JointFeasibleSelector,
    build_joint_feasible_labels,
    compute_stage37_jfi_loss,
    select_stage37_jfi_trajectory,
)


def test_stage37_contract_is_public_initialized_and_sha_locked():
    assert STAGE37_PLAN_SHA256 == (
        "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
    )
    assert STAGE37_ALLOWED_INITIAL_SHA256 == {
        "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
    }
    assert STAGE37_ROUTE_EXPECTED["stage37_gradient_accumulation"] == 8
    assert STAGE37_ROUTE_EXPECTED["stage37_global_bucket_composition"] == (
        2, 30, 8, 24
    )
    assert STAGE37_ROUTE_EXPECTED["stage37_projection_recovery_coefficient"] == 0.25


def test_projection_removes_conflict_without_recovery():
    reward = (torch.tensor([1.0, 0.0]),)
    preserve = (torch.tensor([-1.0, 1.0]),)
    projected, metrics = project_reward_gradient(
        reward, preserve, frontier_delta=0.0
    )
    assert metrics["conflict"].item() == 1.0
    assert metrics["recovery_active"].item() == 0.0
    assert torch.dot(projected[0], preserve[0]).abs() < 1e-6
    assert metrics["finite"].item() == 1.0


def test_projection_adds_exact_recovery_on_frontier_regression():
    reward = (torch.tensor([1.0, 0.0]),)
    preserve = (torch.tensor([0.0, 2.0]),)
    projected, metrics = project_reward_gradient(
        reward, preserve, frontier_delta=-0.0001001
    )
    assert metrics["conflict"].item() == 0.0
    assert metrics["recovery_active"].item() == 1.0
    assert torch.allclose(projected[0], torch.tensor([1.0, 0.5]))


def _selector_config():
    return SimpleNamespace(
        stage37_jfi_num_members=8,
        stage24_selector_dim=128,
    )


def test_jfi_head_quantiles_are_monotonic():
    selector = Stage37JointFeasibleSelector(_selector_config())
    output = selector(torch.randn(2, 20, 8, 128))
    quantiles = output["delta_quantiles"]
    assert quantiles.shape == (2, 20, 8, 3)
    assert torch.all(quantiles[..., 0] <= quantiles[..., 1])
    assert torch.all(quantiles[..., 1] <= quantiles[..., 2])


def test_joint_label_requires_reward_and_all_three_safety_components():
    rewards = torch.zeros(1, 20)
    components = torch.ones(1, 20, 6)
    valid = torch.ones(1, 20, dtype=torch.bool)
    fallback = torch.tensor([0])
    rewards[0, 1] = 0.006
    rewards[0, 2] = 0.006
    components[0, 2, 1] = 0.9994
    labels = build_joint_feasible_labels(
        rewards=rewards,
        components=components,
        valid_mask=valid,
        fallback_modes=fallback,
    )
    assert labels["joint"][0, 1]
    assert not labels["joint"][0, 2]
    assert not labels["joint"][0, 0]


def test_jfi_loss_backpropagates_only_heads():
    selector = Stage37JointFeasibleSelector(_selector_config())
    embedding = torch.randn(2, 20, 8, 128)
    output = selector(embedding)
    rewards = torch.linspace(0.0, 0.2, 20).repeat(2, 1)
    components = torch.ones(2, 20, 6)
    predictions = {
        "stage37_jfi_joint_logits": output["joint_logits"],
        "stage37_jfi_delta_quantiles": output["delta_quantiles"],
        "raw_rewards": rewards,
        "component_scores": components,
        "reward_valid_mask": torch.ones(2, 20, dtype=torch.bool),
        "stage24_fallback_mode": torch.zeros(2, dtype=torch.long),
        "stage24_member_training_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    losses = compute_stage37_jfi_loss(predictions)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in selector.parameters()
    )
    assert losses["stage37_jfi_quantile_monotonic"].item() == 1.0


def test_jfi_deployment_caps_pool_and_ranks_by_q10():
    logits = torch.zeros(1, 20)
    probabilities = torch.zeros(1, 20, 8)
    probabilities[:, 1:8] = 0.99
    quantiles = torch.zeros(1, 20, 8, 3)
    quantiles[:, 1:8, :, 0] = torch.arange(1, 8).view(1, 7, 1) / 100
    embedding = torch.zeros(1, 20, 128)
    selected, diagnostics = select_stage37_jfi_trajectory(
        reference_logits=logits,
        joint_probabilities=probabilities,
        delta_quantiles=quantiles,
        embedding_mean=embedding,
        joint_threshold=0.9,
        q10_floor=0.0,
        ood_mean=torch.zeros(128),
        ood_variance=torch.ones(128),
        ood_threshold=1.0,
        max_candidates=4,
    )
    assert selected.item() == 7
    assert diagnostics["candidate_pool_size"].item() == 4
    assert diagnostics["eligible"].sum().item() == 3
