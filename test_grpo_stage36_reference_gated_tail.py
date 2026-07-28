from types import SimpleNamespace

import pytest
import torch

from navsim.agents.diffusiondrive.stage35_counterfactual import (
    counterfactual_donor_indices,
)
from navsim.agents.diffusiondrive.stage36_loss_bridge import (
    compute_stage36_transfuser_loss,
)
from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    build_reference_gated_tail_targets,
    compute_reference_gated_tail_ncd_objective,
)


def _banks(batch: int = 1):
    groups, modes = 8, 20
    public = torch.full((batch, groups, modes), 0.80)
    public += torch.linspace(0.0, 0.019, modes).view(1, 1, modes)
    current = public.clone()
    current[:, :, 15:] -= 0.002
    current[:, 0:2, 3] += 0.004
    valid = torch.ones_like(current, dtype=torch.bool)
    components = torch.ones(batch, groups, modes, 6)
    return current, public, valid, components


def _selection(batch: int = 1):
    active_modes = torch.arange(4).view(1, 1, 4).expand(
        batch, 8, 4
    ).clone()
    selected = torch.zeros(batch, 8, dtype=torch.long)
    return {
        "current_selected_modes": selected,
        "public_selected_modes": selected.clone(),
        "active_modes": active_modes,
        "active_valid": torch.ones_like(active_modes, dtype=torch.bool),
        "hybrid_selected_modes": torch.zeros(
            batch, 8, 4, 7, dtype=torch.long
        ),
        "donor_indices": counterfactual_donor_indices(8),
    }


def test_stage36_targets_use_real_public_top5_and_same_anchor_top2():
    current, public, valid, components = _banks()
    targets = build_reference_gated_tail_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=public,
        base_valid_mask=valid,
        base_component_scores=components,
        scene_buckets=torch.tensor([0]),
    )
    assert targets["public_frontier_modes"].tolist() == [
        [[19, 18, 17, 16, 15]] * 8
    ]
    assert targets["retention_active"].all()
    assert targets["tail_group_indices"][0, 3].tolist() == [1, 0]
    assert targets["tail_valid"].all()
    assert targets["tail_advantages"][0, 3].gt(0).all()


def test_stage36_safety_regression_overrides_tail_gain():
    current, public, valid, components = _banks()
    current[:, 0:2, 3] += 0.02
    current_components = components.clone()
    current_components[:, 0:2, 3, 0] = 0.5
    targets = build_reference_gated_tail_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=current_components,
        base_rewards=public,
        base_valid_mask=valid,
        base_component_scores=components,
        scene_buckets=torch.tensor([0]),
    )
    assert targets["tail_safety_regression"][0, 3].all()
    assert targets["tail_advantages"][0, 3].eq(-2.0).all()


def test_stage36_objective_is_finite_and_reaches_all_three_terms():
    current, public, valid, components = _banks()
    targets = build_reference_gated_tail_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=public,
        base_valid_mask=valid,
        base_component_scores=components,
        scene_buckets=torch.tensor([0]),
    )
    selection = _selection()
    steps = 5
    current_logp = torch.zeros(1, 8, 4, steps, requires_grad=True)
    bc_logp = torch.full_like(current_logp, -2.0, requires_grad=True)
    ncd_kl = torch.full_like(current_logp, 0.5, requires_grad=True)
    tail_logp = torch.zeros(1, 20, 2, steps, requires_grad=True)
    tail_kl = torch.full_like(tail_logp, 0.5, requires_grad=True)
    frontier_bc = torch.full(
        (1, 8, 5, steps), -2.0, requires_grad=True
    )
    objective = compute_reference_gated_tail_ncd_objective(
        current_log_probs=current_logp,
        bc_log_probs=bc_logp,
        reference_mean_kl=ncd_kl,
        tail_current_log_probs=tail_logp,
        tail_reference_mean_kl=tail_kl,
        frontier_bc_log_probs=frontier_bc,
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=public,
        base_valid_mask=valid,
        base_component_scores=components,
        scene_buckets=torch.tensor([0]),
        public_frontier_modes=targets["public_frontier_modes"],
        public_frontier_valid=targets["public_frontier_valid"],
        retention_coefficients=targets["retention_coefficients"],
        tail_group_indices=targets["tail_group_indices"],
        tail_valid=targets["tail_valid"],
        **selection,
    )
    assert torch.isfinite(objective["loss"])
    assert objective["frontier_retention_active_fraction"].item() > 0
    assert objective["tail_expansion_active_fraction"].item() > 0
    objective["loss"].backward()
    assert bc_logp.grad.abs().sum().item() > 0
    assert frontier_bc.grad.abs().sum().item() > 0
    assert tail_logp.grad.abs().sum().item() > 0
    assert tail_kl.grad.abs().sum().item() > 0


def test_stage36_loss_bridge_rejects_accounting_drift():
    current, public, valid, components = _banks()
    targets = build_reference_gated_tail_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=public,
        base_valid_mask=valid,
        base_component_scores=components,
        scene_buckets=torch.tensor([0]),
    )
    selection = _selection()
    predictions = {
        "diffgrpo_current_log_probs": torch.zeros(1, 8, 4, 5),
        "diffgrpo_bc_log_probs": torch.zeros(1, 8, 4, 5),
        "diffgrpo_reference_mean_kl": torch.zeros(1, 8, 4, 5),
        "stage36_tail_current_log_probs": torch.zeros(1, 20, 2, 5),
        "stage36_tail_reference_mean_kl": torch.zeros(1, 20, 2, 5),
        "stage36_frontier_bc_log_probs": torch.zeros(1, 8, 5, 5),
        "raw_rewards": current.reshape(1, 160),
        "reward_valid_mask": valid.reshape(1, 160),
        "component_scores": components.reshape(1, 160, 6),
        "diffgrpo_base_rewards": public.reshape(1, 160),
        "diffgrpo_base_valid_mask": valid.reshape(1, 160),
        "diffgrpo_base_component_scores": components.reshape(1, 160, 6),
        "stage36_scene_buckets": torch.tensor([0]),
        "stage36_public_frontier_modes": targets["public_frontier_modes"],
        "stage36_public_frontier_valid": targets["public_frontier_valid"],
        "stage36_retention_coefficients": targets["retention_coefficients"],
        "stage36_tail_group_indices": targets["tail_group_indices"],
        "stage36_tail_valid": targets["tail_valid"],
        "stage36_current_unique_replay_count": torch.tensor(40),
        "stage36_public_unique_replay_count": torch.tensor(60),
        "stage36_current_physical_replay_count": torch.tensor(40),
        "stage36_public_physical_replay_count": torch.tensor(64),
        "stage36_reward_dependent_replay": torch.tensor(1),
        "stage36_replay_deduplicated": torch.tensor(1),
        "stage36_sampled_chain_count": torch.tensor(320),
        "stage36_counterfactual_selector_count": torch.tensor(224),
        "stage36_selector_mode_disagreement": torch.tensor(0.0),
        "stage36_current_selector_switch_rate": torch.tensor(0.0),
        "stage36_public_selector_switch_rate": torch.tensor(0.0),
        "diffgrpo_group_size": torch.tensor(8),
        **{
            f"stage36_{name}": value
            for name, value in selection.items()
        },
    }
    result = compute_stage36_transfuser_loss(predictions, SimpleNamespace())
    assert torch.isfinite(result["loss"])
    predictions["stage36_sampled_chain_count"] = torch.tensor(319)
    with pytest.raises(ValueError, match="accounting drifted"):
        compute_stage36_transfuser_loss(predictions, SimpleNamespace())
