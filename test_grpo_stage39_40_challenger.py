import torch

from navsim.agents.diffusiondrive.stage39_challenger import (
    build_stage39_challenger_targets,
    compute_stage39_challenger_objective,
)
from scripts.evaluation.summarize_grpo_stage39 import record_metrics
from navsim.agents.diffusiondrive.stage40_hierarchical_selector import (
    Stage40HierarchicalSelector,
    build_stage40_relative_features,
    select_stage40_hierarchical,
)


def _stage39_banks(batch=2):
    public = torch.arange(20, dtype=torch.float32).view(1, 1, 20) / 100.0
    public = public.expand(batch, 8, 20).clone()
    challenger = public.clone()
    valid = torch.ones_like(public, dtype=torch.bool)
    components = torch.ones(batch, 8, 20, 6)
    return challenger, valid, components, public


def test_stage39_set_credit_measures_public_frontier_marginal():
    challenger, valid, components, public = _stage39_banks()
    challenger[..., 14] = 0.99
    target = build_stage39_challenger_targets(
        challenger_rewards=challenger,
        challenger_valid_mask=valid,
        challenger_component_scores=components,
        public_rewards=public,
        public_valid_mask=valid,
        public_component_scores=components,
    )
    assert target["public_elite_modes"][0, 0].tolist() == [19, 18, 17, 16, 15]
    assert float(target["set_delta"][0, 0, 14]) > 0.1
    assert float(target["set_advantages"][0, 0, 14]) > 0.0
    assert float(target["union_safe_oracle_gain"].mean()) > 0.1


def test_stage39_safety_veto_overrides_high_reward():
    challenger, valid, components, public = _stage39_banks()
    challenger[..., 14] = 0.99
    components = components.clone()
    components[..., 14, 0] = 0.0
    target = build_stage39_challenger_targets(
        challenger_rewards=challenger,
        challenger_valid_mask=valid,
        challenger_component_scores=components,
        public_rewards=public,
        public_valid_mask=valid,
        public_component_scores=torch.ones_like(components),
    )
    assert bool(target["safety_regression"][0, 0, 14])
    assert float(target["set_delta"][0, 0, 14]) == 0.0
    assert float(target["set_advantages"][0, 0, 14]) == -2.0


def test_stage39_branch_gradients_are_separated():
    challenger, valid, components, public = _stage39_banks(batch=1)
    group_offset = torch.arange(8, dtype=challenger.dtype).view(1, 8, 1) * 0.002
    challenger = challenger + group_offset
    challenger[..., 14] = 0.99 + group_offset[..., 0]
    for branch in ("BC", "STD", "SET"):
        logp = torch.zeros(1, 8, 20, 5, requires_grad=True)
        bc_logp = torch.zeros(1, 8, 20, 5, requires_grad=True)
        # Positive KL tensor deliberately depends on logp so GRPO branches have
        # a second differentiable trust term without affecting BC attribution.
        kl = 0.01 * logp.square()
        result = compute_stage39_challenger_objective(
            branch=branch,
            challenger_log_probs=logp,
            challenger_reference_kl=kl,
            public_bc_log_probs=bc_logp,
            challenger_rewards=challenger,
            challenger_valid_mask=valid,
            challenger_component_scores=components,
            public_rewards=public,
            public_valid_mask=valid,
            public_component_scores=components,
        )
        result["loss"].backward()
        if branch == "BC":
            assert bc_logp.grad is not None and float(bc_logp.grad.abs().sum()) > 0
            assert logp.grad is None or float(logp.grad.abs().sum()) == 0
        else:
            assert logp.grad is not None and float(logp.grad.abs().sum()) > 0
            assert bc_logp.grad is None


def _stage40_bank(batch, modes):
    trajectory = torch.zeros(batch, modes, 8, 3)
    trajectory[:, :, :, 0] = torch.linspace(0.0, 1.0, 8)
    trajectory[:, :, :, 1] = torch.arange(modes).view(1, modes, 1) * 0.01
    logits = torch.linspace(1.0, 0.0, modes).view(1, modes).expand(batch, modes)
    fallback = torch.zeros(batch, dtype=torch.long)
    return trajectory, logits, fallback


def test_stage40_features_are_candidate_count_agnostic():
    for modes in (12, 40):
        trajectory, logits, fallback = _stage40_bank(2, modes)
        features = build_stage40_relative_features(trajectory, fallback, logits)
        assert features.shape == (2, modes, 12)
        selector = Stage40HierarchicalSelector(hidden_dim=16, ensemble_size=2)
        outputs = selector(trajectory, fallback, logits)
        assert outputs["delta_mean"].shape == (2, modes)
        assert outputs["component_delta_mean"].shape == (2, modes, 6)


def test_stage40_fail_closed_to_exact_public_fallback():
    trajectory, logits, fallback = _stage40_bank(1, 6)
    outputs = {
        "delta_mean": torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        "delta_std": torch.zeros(1, 6),
        "component_delta_mean": torch.zeros(1, 6, 6),
        "component_delta_std": torch.zeros(1, 6, 6),
        "risk_logits": torch.full((1, 6, 2), 20.0),
        "risk_std": torch.zeros(1, 6, 2),
    }
    selected, diagnostics = select_stage40_hierarchical(
        fallback, logits, outputs
    )
    assert selected.tolist() == fallback.tolist()
    assert not bool(diagnostics["switch"].item())


def test_stage39_summarizer_strict_safe_oracle_and_catastrophe():
    rewards = [i / 100.0 for i in range(20)] + [None] * 20
    rewards[20] = 0.99
    components = [[1.0] * 6 for _ in range(40)]
    components[21][0] = 0.0
    rewards[21] = 0.98
    record = {
        "candidate_rewards": rewards,
        "candidate_components": components,
        "selected_reward": 0.19,
    }
    metrics = record_metrics(record)
    assert metrics["safe_union_gain"] > 0.1
    assert metrics["positive_fraction"] > 0.0
    assert metrics["catastrophic_rate"] == 19 / 20
