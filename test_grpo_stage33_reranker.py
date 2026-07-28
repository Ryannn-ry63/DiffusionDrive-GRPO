import torch

from navsim.agents.diffusiondrive.stage33_cdc_grpo import compute_cdc_advantages
from navsim.agents.diffusiondrive.transfuser_loss import compute_stage33_cdc_smoke_objective
from navsim.agents.diffusiondrive.stage33_external_reranker import (
    FEATURE_DIM,
    Stage33ExternalReranker,
    build_candidate_feature_matrix,
    compute_stage33_reranker_loss,
    select_stage33,
)


def _record():
    return {
        "candidate_trajectories": torch.randn(20, 8, 3).tolist(),
        "candidate_reference_logits": torch.randn(20).tolist(),
        "selector_probabilities": torch.softmax(torch.randn(20), dim=0).tolist(),
        "candidate_rewards": torch.rand(20).tolist(),
        "delta_predictions": torch.randn(20, 4).tolist(),
        "risk_ucb": torch.rand(20).tolist(),
        "delta_lcb": torch.randn(20).tolist(),
        "ood_distance": torch.rand(20).tolist(),
        "harm_probabilities": torch.rand(20, 3).tolist(),
        "catastrophe_probabilities": torch.rand(20, 2).tolist(),
    }


def test_feature_matrix_is_fixed_and_label_free():
    record = _record()
    features = build_candidate_feature_matrix(record)
    assert features.shape == (20, FEATURE_DIM)
    assert torch.isfinite(features).all()
    record.pop("candidate_rewards")
    record.pop("candidate_components", None)
    assert torch.isfinite(build_candidate_feature_matrix(record)).all()


def test_missing_optional_diagnostics_are_zero_filled():
    record = _record()
    for key in (
        "delta_predictions",
        "risk_ucb",
        "delta_lcb",
        "ood_distance",
        "harm_probabilities",
        "catastrophe_probabilities",
    ):
        record.pop(key)
    features = build_candidate_feature_matrix(record)
    assert features.shape == (20, FEATURE_DIM)
    assert torch.all(features[:, -1] == 0)


def test_reranker_loss_has_gradients():
    model = Stage33ExternalReranker(hidden_dim=32, num_layers=1)
    outputs = model(torch.randn(3, 20, FEATURE_DIM))
    result = compute_stage33_reranker_loss(outputs, torch.rand(3, 20))
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_selection_mask_and_fallback():
    outputs = {
        "score_mean": torch.tensor([[0.1, 3.0] + [0.0] * 18]),
        "score_log_std": torch.zeros(1, 20),
        "catastrophe_logit": torch.zeros(1, 20),
    }
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, 0] = True
    result = select_stage33(outputs, eligible_mask=mask, fallback_mode=7)
    assert int(result["selected_mode"][0]) == 0
    result = select_stage33(outputs, eligible_mask=torch.zeros_like(mask), fallback_mode=7)
    assert int(result["selected_mode"][0]) == 7
    assert bool(result["fallback_used"][0])


def test_cdc_credit_is_selected_and_probability_weighted():
    current = torch.tensor([[0.8, 0.5, 0.6]])
    public = torch.tensor([[0.7, 0.5, 0.55]])
    result = compute_cdc_advantages(
        current,
        public,
        current_selected_modes=torch.tensor([0]),
        public_selected_modes=torch.tensor([0]),
        selector_probabilities=torch.tensor([[0.8, 0.1, 0.1]]),
    )
    assert result["advantages"].shape == (1, 3)
    assert torch.isfinite(result["advantages"]).all()
    assert result["deployment_delta"].item() > 0


def test_cdc_smoke_objective_is_finite():
    shape = (2, 8, 5)
    result = compute_stage33_cdc_smoke_objective(
        current_log_probs=torch.randn(*shape),
        bc_log_probs=torch.randn(*shape),
        reference_mean_kl=torch.rand(*shape).abs(),
        rewards=torch.rand(2, 8),
        valid_mask=torch.ones(2, 8, dtype=torch.bool),
        component_scores=torch.rand(2, 8, 6),
        base_rewards=torch.rand(2, 8),
        base_valid_mask=torch.ones(2, 8, dtype=torch.bool),
        base_component_scores=torch.rand(2, 8, 6),
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["stage33_deployment_credit_mean"])
