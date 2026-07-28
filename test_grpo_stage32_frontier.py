import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _gather_set_mode_pool,
    _stage32_frontier_pool,
)
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_selector_aware_frontier_objective,
)


def test_stage32_pool_excludes_fallback_and_selected():
    batch, groups, modes, pool = 2, 8, 20, 4
    diagnostics = {
        "fallback_mode": torch.zeros(batch * groups, dtype=torch.long),
        "risk_ucb": torch.zeros(batch * groups, modes),
        "ood_distance": torch.zeros(batch * groups, modes),
        "delta_lcb": torch.arange(modes, dtype=torch.float32).repeat(batch * groups, 1),
    }
    selected = torch.ones(batch * groups, dtype=torch.long)
    indices, valid = _stage32_frontier_pool(
        diagnostics, selected, groups, modes, 1.0, 0.1, 0.1, pool
    )
    assert indices.shape == (batch, groups, pool + 1)
    assert valid.all()
    assert (indices[..., 0] == 1).all()
    assert not (indices[..., 1:] == 0).any()
    assert not (indices[..., 1:] == 1).any()


def test_stage32_pool_gather_preserves_group_layout():
    batch, groups, modes, pool = 2, 8, 20, 5
    values = torch.arange(batch * groups * modes * 3).reshape(batch, groups * modes, 3)
    indices = torch.randint(0, modes, (batch, groups, pool))
    gathered = _gather_set_mode_pool(values, indices, modes)
    assert gathered.shape == (batch, groups * pool, 3)


def test_stage32_objective_is_finite_and_has_gradient():
    batch, groups, pool, steps = 2, 8, 4, 5
    total = groups * (pool + 1)
    current = torch.randn(batch, total, steps, requires_grad=True)
    teacher = torch.randn(batch, total, steps)
    exact_kl = torch.rand(batch, total, steps)
    rewards = torch.rand(batch, total)
    base_rewards = rewards - 0.01 * torch.randn(batch, total)
    valid = torch.ones(batch, total, dtype=torch.bool)
    components = torch.rand(batch, total, 6)
    current_valid = torch.ones(batch, total, dtype=torch.bool)
    public_valid = torch.ones(batch, total, dtype=torch.bool)
    result = compute_selector_aware_frontier_objective(
        current, teacher, exact_kl, rewards, valid, components,
        base_rewards, valid, components.clone(), current_valid, public_valid,
        group_size=groups, pool_size=pool,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_stage32_bc_and_kl_are_normalized_over_valid_replay_chains():
    batch, groups, pool, steps = 2, 8, 4, 5
    total = groups * (pool + 1)
    current = torch.zeros(batch, total, steps, requires_grad=True)
    teacher = torch.full((batch, total, steps), -2.0)
    exact_kl = torch.full((batch, total, steps), 0.5)
    rewards = torch.full((batch, total), 0.5)
    valid = torch.ones(batch, total, dtype=torch.bool)
    components = torch.ones(batch, total, 6)
    result = compute_selector_aware_frontier_objective(
        current, teacher, exact_kl, rewards, valid, components,
        rewards.clone(), valid, components.clone(), valid, valid,
        group_size=groups, pool_size=pool, step_discount=1.0,
        bc_weight=0.1, kl_weight=0.1, safety_kl_weight=0.5,
    )
    # Mean(-log p) * BC weight and mean(KL) * KL weight.  Neither may
    # scale with the 40-chain replay width or the five diffusion steps.
    assert torch.allclose(result["bc_loss"], torch.tensor(0.2))
    assert torch.allclose(result["reference_kl_loss"], torch.tensor(0.05))
    assert torch.allclose(result["loss"], torch.tensor(0.25))
