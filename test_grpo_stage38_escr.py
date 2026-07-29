import torch

from navsim.agents.diffusiondrive.stage38_elite_set_repair import (
    build_elite_set_repair_targets,
    compute_elite_set_repair_objective,
)


def _banks():
    base = torch.arange(20, dtype=torch.float32).view(1, 1, 20) / 100.0
    base = base.expand(2, 8, 20).clone()
    current = base.clone()
    valid = torch.ones(2, 8, 20, dtype=torch.bool)
    components = torch.ones(2, 8, 20, 6)
    buckets = torch.tensor([3, 0])
    selected = torch.zeros(2, 8, dtype=torch.long)
    return current, valid, components, base, buckets, selected


def _targets(current, valid, components, base, buckets, selected):
    return build_elite_set_repair_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=base,
        base_valid_mask=valid,
        base_component_scores=components,
        current_selected_modes=selected,
        scene_buckets=buckets,
    )


def test_public_top5_is_deterministic_and_zero_on_identical_policy():
    current, valid, components, base, buckets, selected = _banks()
    target = _targets(current, valid, components, base, buckets, selected)
    assert target["public_elite_modes"][0, 0].tolist() == [19, 18, 17, 16, 15]
    assert torch.equal(target["active_modes"][0, 0, :5], target["public_elite_modes"][0, 0])
    assert float(target["delta_set"].abs().max()) == 0.0
    assert float(target["union_safe_oracle_gain"].abs().max()) == 0.0


def test_non_elite_positive_challenger_enters_active_pool():
    current, valid, components, base, buckets, selected = _banks()
    current[..., 14] = 0.99
    target = _targets(current, valid, components, base, buckets, selected)
    assert float(target["delta_set"][0, 0, 14]) > 0.8
    assert bool(target["active_valid"][0, 0, 5])
    assert int(target["active_modes"][0, 0, 5]) == 14


def test_safety_regression_overrides_positive_credit():
    current, valid, components, base, buckets, selected = _banks()
    current[..., 14] = 0.99
    components = components.clone()
    components[..., 14, 0] = 0.0
    target = build_elite_set_repair_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=base,
        base_valid_mask=valid,
        base_component_scores=torch.ones_like(components),
        current_selected_modes=selected,
        scene_buckets=buckets,
    )
    assert bool(target["safety_regression"][0, 0, 14])
    assert float(target["advantages"][0, 0, 14]) == -2.0


def test_invalid_current_elite_gets_maximum_negative_credit_and_replay():
    current, valid, components, base, buckets, selected = _banks()
    valid = valid.clone()
    valid[..., 19] = False
    target = build_elite_set_repair_targets(
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=base,
        base_valid_mask=torch.ones_like(valid),
        base_component_scores=components,
        current_selected_modes=selected,
        scene_buckets=buckets,
    )
    assert bool(target["invalid_elite"][0, 0, 19])
    assert bool(target["target_valid"][0, 0, 19])
    assert bool(target["safety_regression"][0, 0, 19])
    assert float(target["advantages"][0, 0, 19]) == -2.0
    elite_slot = torch.nonzero(
        target["public_elite_modes"][0, 0] == 19, as_tuple=False
    ).item()
    assert bool(target["active_valid"][0, 0, elite_slot])


def test_non_elite_below_public_frontier_never_gets_positive_credit():
    current, valid, components, base, buckets, selected = _banks()
    current[..., 14] = 0.145
    target = _targets(current, valid, components, base, buckets, selected)
    assert float(target["delta_set"][0, 0, 14]) == 0.0
    assert float(target["advantages"][0, 0, 14]) == 0.0
    assert 14 not in target["active_modes"][0, 0, 5:7].tolist()


def test_public_elite_regression_gets_negative_credit():
    current, valid, components, base, buckets, selected = _banks()
    current[..., 19] = 0.10
    target = _targets(current, valid, components, base, buckets, selected)
    assert float(target["delta_set"][0, 0, 19]) < 0.0
    assert float(target["advantages"][0, 0, 19]) < 0.0


def test_mode_permutation_preserves_set_utilities_without_ties():
    current, valid, components, base, buckets, selected = _banks()
    current[..., 14] = 0.99
    original = _targets(current, valid, components, base, buckets, selected)
    permutation = torch.tensor([
        7, 3, 18, 0, 11, 14, 1, 19, 5, 9,
        16, 2, 13, 6, 12, 4, 17, 8, 15, 10,
    ])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(20)
    permuted_selected = inverse[selected]
    permuted = _targets(
        current[..., permutation],
        valid[..., permutation],
        components[..., permutation, :],
        base[..., permutation],
        buckets,
        permuted_selected,
    )
    assert torch.equal(original["public_utility"], permuted["public_utility"])
    assert torch.equal(
        original["delta_set"], permuted["delta_set"][..., inverse]
    )
    assert torch.equal(
        original["union_safe_oracle_gain"],
        permuted["union_safe_oracle_gain"],
    )


def test_objective_has_decoder_and_elite_bc_gradients():
    current, valid, components, base, buckets, selected = _banks()
    current[..., 14] = 0.99
    target = _targets(current, valid, components, base, buckets, selected)
    log_probs = torch.zeros(2, 8, 8, 5, requires_grad=True)
    reference_kl = torch.full((2, 8, 8, 5), 0.01)
    bc_log_probs = torch.zeros(2, 8, 5, 5, requires_grad=True)
    result = compute_elite_set_repair_objective(
        current_log_probs=log_probs,
        reference_mean_kl=reference_kl,
        public_elite_bc_log_probs=bc_log_probs,
        rewards=current,
        valid_mask=valid,
        component_scores=components,
        base_rewards=base,
        base_valid_mask=valid,
        base_component_scores=components,
        current_selected_modes=selected,
        scene_buckets=buckets,
        active_modes=target["active_modes"],
        active_valid=target["active_valid"],
        public_elite_modes=target["public_elite_modes"],
        public_elite_valid=target["public_elite_valid"],
    )
    result["loss"].backward()
    assert log_probs.grad is not None and float(log_probs.grad.abs().sum()) > 0
    assert bc_log_probs.grad is not None and float(bc_log_probs.grad.abs().sum()) > 0
