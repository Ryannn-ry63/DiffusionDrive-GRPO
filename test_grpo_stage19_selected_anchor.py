import json
from dataclasses import replace

import pytest
import torch

from navsim.agents.diffusiondrive import diffusion_grpo
from navsim.agents.diffusiondrive.transfuser_agent import (
    validate_formal_grpo_config,
)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_loss import (
    compute_full_chain_diffgrpo_objective,
)
from navsim.agents.diffusiondrive.transfuser_model_v2 import (
    STAGE19_BASE_SHA256,
    STAGE19_FULL_SCHEDULE,
    load_selected_anchor_modes,
)
from scripts.evaluation.build_grpo_stage19_manifests import assign_logs


class FakeScheduler:
    def add_noise(self, original_samples, noise, timesteps):
        assert timesteps.shape == (original_samples.shape[0],)
        return original_samples + noise


class FakeHead:
    def __init__(self):
        self.ref_policy = object()
        self.diff_decoder = object()
        self.diffusion_scheduler = FakeScheduler()
        self._truncation_timestep = 32
        self._generation_trust_projection_mode = "none"

    def denorm_odo(self, value):
        return value


def test_stage19_trace_uses_eight_same_anchor_rollouts_and_one_teacher(
    monkeypatch,
):
    sampled_shapes = []

    def fake_sample(head, policy, initial, *args):
        sampled_shapes.append(initial.shape)
        states = tuple(initial for _ in range(5))
        actions = tuple(initial for _ in range(5))
        classification = initial.new_zeros(initial.shape[:2])
        return states, actions, initial, classification

    def fake_replay(head, policy, states, actions, *args):
        shape = states[0].shape
        return states[0].new_zeros(shape[0], shape[1], 5), states[0].new_zeros(
            shape[0], shape[1]
        )

    monkeypatch.setattr(diffusion_grpo, "_sample_full_chain_actions", fake_sample)
    monkeypatch.setattr(diffusion_grpo, "_replay_full_chain_actions", fake_replay)
    clean = torch.zeros(2, 1, 8, 3)
    modes = torch.tensor([3, 17])
    trace = diffusion_grpo.collect_selected_anchor_diffgrpo_trace(
        FakeHead(),
        clean,
        modes,
        8,
        *(torch.zeros(2, 1, 1) for _ in range(3)),
        (1, 1),
        torch.zeros(2, 1),
        None,
    )
    assert sampled_shapes == [
        torch.Size([2, 8, 8, 3]),
        torch.Size([2, 1, 8, 3]),
    ]
    assert trace["current_log_probs"].shape == (2, 8, 5)
    assert trace["bc_log_probs"].shape == (2, 1, 5)
    assert trace["trajectories"].shape == (2, 8, 8, 3)
    assert torch.equal(trace["selected_anchor_modes"], modes)


def test_stage19_objective_accepts_single_teacher_per_scene():
    current = torch.zeros(2, 8, 5, requires_grad=True)
    teacher = torch.full((2, 1, 5), -1.5, requires_grad=True)
    rewards = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16
    result = compute_full_chain_diffgrpo_objective(
        current,
        teacher,
        rewards,
        torch.ones_like(rewards, dtype=torch.bool),
    )
    assert result["bc_loss"].item() == pytest.approx(1.5)
    result["loss"].backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert teacher.grad is not None and teacher.grad.abs().sum() > 0


def test_stage19_manifest_loader_fails_closed(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {
        "summary": {
            "base_checkpoint_sha256": STAGE19_BASE_SHA256,
            "schedule": STAGE19_FULL_SCHEDULE,
            "group_definition": "same_token_same_selected_anchor",
            "group_size": 8,
        },
        "records": [
            {"token": "a", "selected_mode": 3},
            {"token": "b", "selected_mode": 17},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_selected_anchor_modes(str(path), 8) == {"a": 3, "b": 17}
    payload["summary"]["group_size"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="group size"):
        load_selected_anchor_modes(str(path), 8)


def test_stage19_log_assignment_is_deterministic_and_disjoint():
    token_logs = {
        f"token-{log}-{index}": f"log-{log}"
        for log in range(30)
        for index in range(1 + log % 5)
    }
    first = assign_logs(token_logs)
    second = assign_logs(dict(reversed(list(token_logs.items()))))
    assert first == second
    assert set(first.values()) == {"fit", "calibration", "test"}
    assert len(first) == 30


def test_stage19_formal_config_locks_selected_anchor_method():
    config = replace(
        TransfuserConfig(),
        grpo_training_mode="diffgrpo_selected_anchor",
        grpo_decoder_gradient_scope="all_layers",
        inference_selector_source="reference",
        grpo_reward_mode="pdms",
        grpo_scene_weight_mode="uniform",
        selection_behavior_weighting="old_policy",
        generation_advantage_mode="group_zscore",
        generation_mode_weighting="uniform",
        generation_trust_projection_mode="none",
        grpo_rollouts_per_mode=1,
        grpo_priority_manifest_path="",
        grpo_priority_sample_fraction=0.0,
        selection_entropy_weight=0.0,
        selection_exploration_floor=0.0,
        selection_rank_loss_weight=0.0,
        selector_consistency_kl_weight=0.0,
        policy_loss_weight=0.0,
        kl_loss_weight=0.0,
        selector_generation_kl_weight=0.0,
        generation_policy_loss_weight=1.0,
        generation_kl_loss_weight=0.0,
        generation_adaptive_kl_enabled=False,
        generation_policy_algorithm="diffgrpo_selected_anchor",
        diffgrpo_bc_weight=0.1,
        diffgrpo_step_discount=0.6,
        diffgrpo_logprob_reduction="mean",
        diffgrpo_group_size=8,
        diffusion_truncation_timestep=32,
        diffusion_roll_timesteps=(32, 24, 16, 8, 0),
        diffusion_scheduler_num_inference_steps=125,
    )
    validate_formal_grpo_config(config)
    with pytest.raises(ValueError, match="group_size"):
        validate_formal_grpo_config(replace(config, diffgrpo_group_size=4))
