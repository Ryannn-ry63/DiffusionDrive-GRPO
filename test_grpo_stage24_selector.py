import hashlib
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from navsim.agents.diffusiondrive.stage24_safety_value_selector import (
    Stage24SafetyValueSelector,
    compute_stage24_selector_loss,
    load_stage24_candidate_banks,
    select_stage24_trajectory,
    stage24_member_training_mask,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from scripts.evaluation.calibrate_grpo_stage24_selector import (
    clopper_pearson_upper,
    whole_log_bootstrap_ci,
)


def _config():
    return SimpleNamespace(
        tf_d_model=32,
        tf_num_head=4,
        lidar_max_y=32.0,
        lidar_max_x=32.0,
        stage24_selector_num_members=8,
        stage24_selector_num_modes=20,
        stage24_selector_dim=32,
    )


def _inputs(batch=1):
    return (
        torch.randn(batch, 20, 8, 3),
        torch.randn(batch, 20),
        torch.randn(batch, 32, 8, 8),
        torch.randn(batch, 6, 32),
        torch.randn(batch, 1, 32),
        torch.randn(batch, 1, 32),
    )


def test_stage24_subbags_are_whole_log_deterministic_and_diverse():
    logs = ("log-a", "log-a", "log-b", "log-c")
    first = stage24_member_training_mask(logs)
    second = stage24_member_training_mask(logs)
    assert torch.equal(first, second)
    assert torch.equal(first[0], first[1])
    assert first.shape == (4, 8)
    assert first.any(dim=1).all()
    assert torch.unique(first, dim=0).shape[0] > 1


def test_stage24_selector_is_permutation_equivariant_without_mode_identity():
    torch.manual_seed(24024)
    selector = Stage24SafetyValueSelector(_config()).eval()
    trajectories, logits, bev, agents, ego, status = _inputs()
    output = selector(trajectories, logits, bev, (8, 8), agents, ego, status)
    permutation = torch.randperm(20)
    moved = selector(
        trajectories[:, permutation], logits[:, permutation], bev, (8, 8),
        agents, ego, status,
    )
    assert torch.allclose(
        output["delta_predictions"][:, permutation],
        moved["delta_predictions"], atol=2e-5, rtol=2e-5,
    )
    assert torch.allclose(
        output["safety_probabilities"][:, permutation],
        moved["safety_probabilities"], atol=2e-5, rtol=2e-5,
    )


def test_stage24_ensemble_has_eight_independently_initialized_members():
    selector = Stage24SafetyValueSelector(_config())
    fingerprints = []
    for member in selector.members:
        flat = torch.cat([parameter.detach().flatten() for parameter in member.parameters()])
        fingerprints.append(float(flat[:4096].sum()))
    assert len(fingerprints) == 8
    assert len(set(fingerprints)) == 8


def test_stage24_selector_depends_on_trajectory():
    torch.manual_seed(24025)
    selector = Stage24SafetyValueSelector(_config()).eval()
    trajectories, logits, bev, agents, ego, status = _inputs()
    original = selector(trajectories, logits, bev, (8, 8), agents, ego, status)
    changed = trajectories.clone()
    changed[:, 7, :, 0] += torch.linspace(0, 12, 8)
    moved = selector(changed, logits, bev, (8, 8), agents, ego, status)
    assert not torch.allclose(
        original["delta_predictions"][:, 7], moved["delta_predictions"][:, 7]
    )


def test_stage24_loss_is_finite_and_backpropagates_all_heads():
    torch.manual_seed(24026)
    safety_logits = torch.randn(2, 20, 8, 3, requires_grad=True)
    any_logits = torch.randn(2, 20, 8, requires_grad=True)
    component_delta = torch.randn(2, 20, 8, 6, requires_grad=True)
    delta = torch.randn(2, 20, 8, requires_grad=True)
    components = torch.ones(2, 20, 6)
    components[:, 4, 0] = 0.0
    components[:, 8, 3] = 0.0
    rewards = torch.rand(2, 20)
    result = compute_stage24_selector_loss({
        "stage24_safety_logits": safety_logits,
        "stage24_any_unsafe_logits": any_logits,
        "stage24_component_delta_predictions": component_delta,
        "stage24_delta_predictions": delta,
        "component_scores": components,
        "raw_rewards": rewards,
        "reward_valid_mask": torch.ones(2, 20, dtype=torch.bool),
        "stage24_member_training_mask": stage24_member_training_mask(("a", "b")),
        "stage24_fallback_mode": torch.tensor([0, 1]),
        "stage24_safety_positive_weights": torch.tensor([10.0, 10.0, 10.0]),
        "stage24_any_unsafe_positive_weight": torch.tensor(10.0),
    })
    assert torch.isfinite(result["loss"])
    assert result["stage24_active_pair_count"] > 0
    assert result["stage24_false_safe_hinge_loss"] >= 0
    result["loss"].backward()
    for tensor in (safety_logits, any_logits, component_delta, delta):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0


def test_stage24_selection_is_safety_first_and_ood_closed():
    reference = torch.arange(20, dtype=torch.float32).flip(0).unsqueeze(0)
    safety = torch.full((1, 20, 8, 3), 0.01)
    any_unsafe = torch.full((1, 20, 8), 0.01)
    component_delta = torch.zeros(1, 20, 8, 6)
    value = torch.zeros(1, 20, 8)
    value[:, 5] = 0.08
    value[:, 7] = 0.12
    value[:, 9] = 0.20
    any_unsafe[:, 9] = 0.9
    embeddings = torch.zeros(1, 20, 4)
    embeddings[:, 7] = 10.0
    selected, diagnostics = select_stage24_trajectory(
        reference, safety, any_unsafe, component_delta, value, embeddings,
        residual_margin=0.01, risk_threshold=0.1,
        ood_mean=torch.zeros(4), ood_variance=torch.ones(4),
        ood_threshold=2.0,
    )
    assert selected.item() == 5
    assert diagnostics["ood_rejected"][0, 7]
    assert diagnostics["risk_ucb"][0, 9] > 0.1


def test_stage24_bank_loader_requires_domain_provenance_and_alignment(tmp_path):
    records = [{
        "token": "t0", "log_name": "l0",
        "candidate_trajectories": [[[0.0, 0.0, 0.0]] * 8] * 20,
        "candidate_reference_logits": [0.0] * 20,
        "candidate_rewards": [0.5] * 20,
        "candidate_components": [[1.0] * 6] * 20,
    }]
    paths = []
    for domain, namespace in (("base", -1), ("stage21", 4)):
        path = tmp_path / f"{domain}.json"
        path.write_text(json.dumps({
            "summary": {
                "generator_domain": domain,
                "evaluation_noise_namespace": namespace,
                "checkpoint_sha256": domain * 8,
                "stores_candidate_trajectories": True,
            },
            "records": records,
        }))
        paths.append(str(path))
    bank, combinations = load_stage24_candidate_banks(paths, ("t0",))
    assert len(bank["t0"]) == 2
    assert combinations[0][:2] == ("base", -1)
    assert bank["t0"][1]["generator_domain"] == "stage21"


def test_stage24_bank_source_wrapper_is_sha_pinned(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "summary": {
            "generator_domain": "base", "evaluation_noise_namespace": -1,
            "checkpoint_sha256": "a" * 64, "stores_candidate_trajectories": True,
        },
        "records": [{
            "token": "t0", "log_name": "l0",
            "candidate_trajectories": [[[0.0, 0.0, 0.0]] * 8] * 20,
            "candidate_reference_logits": [0.0] * 20,
            "candidate_rewards": [0.5] * 20,
            "candidate_components": [[1.0] * 6] * 20,
        }],
    }))
    wrapper = tmp_path / "wrapper.json"
    wrapper.write_text(json.dumps({"summary": {
        "source_artifact": str(source),
        "source_artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator_domain": "base", "evaluation_noise_namespace": -1,
    }}))
    bank, combinations = load_stage24_candidate_banks([str(wrapper)], ("t0",))
    assert bank["t0"][0]["generator_domain"] == "base"
    assert combinations == (("base", -1, "a" * 64),)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        load_stage24_candidate_banks([str(wrapper)], ("t0",))


def test_stage24_cluster_bootstrap_and_catastrophe_bound_are_deterministic():
    values = np.asarray([0.02, 0.03, -0.01, 0.04, 0.05], dtype=np.float64)
    logs = ["log-a", "log-a", "log-b", "log-c", "log-c"]
    first = whole_log_bootstrap_ci(values, logs, samples=2000, seed=24024)
    second = whole_log_bootstrap_ci(values, logs, samples=2000, seed=24024)
    assert first == second
    assert first[0] < float(values.mean()) < first[1]
    assert clopper_pearson_upper(0, 2000) < 0.002
    assert clopper_pearson_upper(1, 2000) > clopper_pearson_upper(0, 2000)


def test_stage24_model_inference_cannot_call_internal_pdm_or_base_fallback():
    model_source = Path(
        "navsim/agents/diffusiondrive/transfuser_model_v2.py"
    ).read_text()
    selector_source = Path(
        "navsim/agents/diffusiondrive/stage24_safety_value_selector.py"
    ).read_text()
    assert '"trajectory_relative_harm_v3",' in model_source
    assert 'not in {' in model_source
    assert "_compute_rewards_from_lazy_cache" not in selector_source
    assert "ref_policy" not in selector_source


def test_stage24_training_manifest_is_filtered_and_whole_log_isolated():
    train = json.loads(Path(
        "artifacts/grpo_stage24/manifests/selector_train_manifest.json"
    ).read_text())
    calibration = json.loads(Path(
        "artifacts/grpo_stage24/manifests/selector_calibration_manifest.json"
    ).read_text())
    assert train["summary"]["source_folds"] == [0, 1, 2]
    assert train["summary"]["count"] == len(train["records"]) == 3056
    assert calibration["summary"]["source_folds"] == [3]
    assert calibration["summary"]["count"] == len(calibration["records"]) == 1019
    train_logs = {record["log_name"] for record in train["records"]}
    calibration_logs = {
        record["log_name"] for record in calibration["records"]
    }
    assert len(train_logs) == 454
    assert len(calibration_logs) == 152
    assert train_logs.isdisjoint(calibration_logs)
    training_source = Path("navsim/planning/script/run_training.py").read_text()
    assert '"stage24_selector", "stage25_relative_harm_selector"' in training_source
    assert "Stage24 manifest filtering changed token count" in training_source


def test_stage24_dataclass_and_hydra_schema_are_in_sync():
    yaml_config = OmegaConf.load(
        "navsim/planning/script/config/common/agent/diffusiondrive_agent.yaml"
    )["config"]
    dataclass_names = {
        field.name for field in fields(TransfuserConfig)
        if field.name.startswith("stage24_")
    }
    yaml_names = {
        name for name in yaml_config.keys() if name.startswith("stage24_")
    }
    assert yaml_names == dataclass_names


def test_stage24_selector_falls_back_without_positive_lcb():
    reference = torch.randn(2, 20)
    safety = torch.zeros(2, 20, 8, 3)
    any_unsafe = torch.zeros(2, 20, 8)
    component_delta = torch.zeros(2, 20, 8, 6)
    value = torch.zeros(2, 20, 8)
    embedding = torch.zeros(2, 20, 4)
    selected, diagnostics = select_stage24_trajectory(
        reference, safety, any_unsafe, component_delta, value, embedding,
        residual_margin=0.01, risk_threshold=0.1,
        ood_mean=torch.zeros(4), ood_variance=torch.ones(4), ood_threshold=1.0,
    )
    assert torch.equal(selected, reference.argmax(dim=-1))
    assert not diagnostics["switch"].any()
