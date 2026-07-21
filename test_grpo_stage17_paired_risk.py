from dataclasses import replace

import pytest
import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    validate_inference_selector_source,
)
from navsim.agents.diffusiondrive.paired_advantage_risk import (
    PAIR_EVENT_NAMES,
    PairedAdvantageRiskHead,
    build_paired_advantage_risk_targets,
    compute_paired_advantage_risk_loss,
    select_same_mode_with_base_fallback,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from scripts.evaluation.merge_grpo_stage17_artifacts import (
    validate_shard_provenance,
)


def paired_targets():
    current = torch.tensor([[0.8, 0.0, 0.4]])
    base = torch.tensor([[0.7, 0.9, 0.4]])
    current_components = torch.ones(1, 3, 6)
    base_components = torch.ones(1, 3, 6)
    current_components[0, 1, 0] = 0.0
    current_components[0, 1, 3] = 0.0
    valid = torch.ones(1, 3, dtype=torch.bool)
    return current, base, current_components, base_components, valid


def test_stage17_labels_encode_advantage_tail_and_safety():
    current, base, current_components, base_components, valid = paired_targets()
    result = build_paired_advantage_risk_targets(
        current, base, current_components, base_components, valid, valid
    )
    assert result["labels"].shape == (1, 3, len(PAIR_EVENT_NAMES))
    assert result["labels"][0, 0].tolist() == [0, 0, 0, 0, 0, 0]
    assert result["labels"][0, 1].tolist() == [1, 1, 1, 1, 0, 1]
    assert result["labels"][0, 2].tolist() == [0, 0, 0, 0, 0, 0]
    assert torch.allclose(result["delta"], torch.tensor([[0.1, -0.9, 0.0]]))


def test_stage17_loss_updates_only_prediction_tensors():
    current, base, current_components, base_components, valid = paired_targets()
    logits = torch.zeros(1, 3, 6, requires_grad=True)
    delta = torch.zeros(1, 3, requires_grad=True)
    predictions = {
        "paired_risk_event_logits": logits,
        "paired_risk_delta_prediction": delta,
        "paired_current_rewards": current,
        "paired_base_rewards": base,
        "paired_current_components": current_components,
        "paired_base_components": base_components,
        "paired_current_valid": valid,
        "paired_base_valid": valid,
        "paired_reference_logits": torch.tensor([[0.0, 2.0, 1.0]]),
    }
    result = compute_paired_advantage_risk_loss(
        predictions, positive_weights=(1, 2, 3, 4, 5, 6)
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert delta.grad is not None and delta.grad.abs().sum() > 0
    assert result["paired_risk_loss_0p5_rate"].item() == pytest.approx(1 / 3)


def test_stage17_exact_same_mode_fallback():
    current = torch.arange(2 * 3 * 8 * 3, dtype=torch.float32).reshape(2, 3, 8, 3)
    base = current + 1000
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 1.0]])
    scores = torch.tensor([[0.1, 0.9, 0.2], [0.2, 0.9, 0.1]])
    selected, diagnostics = select_same_mode_with_base_fallback(
        current, base, logits, scores, threshold=0.5
    )
    assert diagnostics["mode"].tolist() == [1, 0]
    assert diagnostics["fallback"].tolist() == [True, False]
    assert torch.equal(selected[0], base[0, 1])
    assert torch.equal(selected[1], current[1, 0])


def test_stage17_head_shapes_and_gradient():
    config = replace(TransfuserConfig(), tf_d_model=256, tf_num_head=8)
    head = PairedAdvantageRiskHead(config)
    current = torch.randn(2, 3, 8, 3)
    base = torch.randn_like(current)
    bev = torch.randn(2, 256, 8, 8)
    agents = torch.randn(2, 4, 256)
    ego = torch.randn(2, 1, 256)
    status = torch.randn(2, 256)
    output = head(current, base, bev, (8, 8), agents, ego, status)
    assert output["event_logits"].shape == (2, 3, 6)
    assert output["delta_prediction"].shape == (2, 3)
    output["event_logits"].sum().backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_stage17_selector_source_is_explicitly_trajectory_conditioned():
    assert validate_inference_selector_source("paired_tail_risk") == "paired_tail_risk"
    with pytest.raises(ValueError, match="resolved by TrajectoryHead"):
        from navsim.agents.diffusiondrive.diffusion_grpo import select_inference_mode

        select_inference_mode(
            torch.zeros(1, 2), torch.zeros(1, 2), "paired_tail_risk"
        )


def test_stage17_shard_merge_rejects_mixed_adapter_provenance():
    common = {
        "checkpoint_sha256": "generator",
        "reference_checkpoint_sha256": "reference",
        "selector_logits_source": "paired_tail_risk",
        "schedule": [32, 24, 16, 8, 0],
        "paired_risk": {
            "checkpoint_sha256": "adapter-a",
            "threshold": 0.5,
            "calibration_path": "calibration.json",
        },
    }
    validate_shard_provenance([common, common])
    mixed = {
        **common,
        "paired_risk": {**common["paired_risk"], "checkpoint_sha256": "adapter-b"},
    }
    with pytest.raises(RuntimeError, match="paired-risk provenance mismatch"):
        validate_shard_provenance([common, mixed])
