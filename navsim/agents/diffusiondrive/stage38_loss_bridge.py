"""Loss bridge and trace-accounting checks for Stage38 ESCR."""

from __future__ import annotations

from typing import Any, Dict

import torch

from navsim.agents.diffusiondrive.stage38_elite_set_repair import (
    compute_elite_set_repair_objective,
)


def _bank(
    value: torch.Tensor,
    *,
    batch: int,
    groups: int,
    modes: int,
    trailing: tuple[int, ...] = (),
) -> torch.Tensor:
    expected = (batch, groups * modes, *trailing)
    if value.shape != expected:
        raise ValueError(
            f"Stage38 bank tensor must have shape {expected}, got {value.shape}"
        )
    return value.reshape(batch, groups, modes, *trailing)


def compute_stage38_transfuser_loss(
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Validate ESCR compact replay and expose stable training diagnostics."""
    required = (
        "diffgrpo_current_log_probs",
        "diffgrpo_reference_mean_kl",
        "stage38_public_elite_bc_log_probs",
        "raw_rewards",
        "reward_valid_mask",
        "component_scores",
        "diffgrpo_base_rewards",
        "diffgrpo_base_valid_mask",
        "diffgrpo_base_component_scores",
        "stage38_current_selected_modes",
        "stage38_scene_buckets",
        "stage38_active_modes",
        "stage38_active_valid",
        "stage38_public_elite_modes",
        "stage38_public_elite_valid",
        "stage38_delta_set",
        "stage38_escr_advantages",
        "stage38_union_safe_oracle_gain",
        "stage38_current_unique_replay_count",
        "stage38_public_unique_replay_count",
        "stage38_current_physical_replay_count",
        "stage38_public_physical_replay_count",
        "stage38_reward_dependent_replay",
        "stage38_replay_deduplicated",
        "stage38_sampled_chain_count",
        "stage38_counterfactual_selector_count",
        "diffgrpo_group_size",
    )
    missing = [name for name in required if predictions.get(name) is None]
    if missing:
        raise ValueError(f"Missing Stage38 ESCR tensors: {missing}")

    current_log_probs = predictions["diffgrpo_current_log_probs"]
    if current_log_probs.ndim != 4:
        raise ValueError("Stage38 current log probabilities must be [B,8,8,S]")
    batch, groups, active_width, steps = current_log_probs.shape
    elite_width = int(getattr(config, "stage38_elite_width", 5))
    if (groups, active_width, elite_width) != (8, 8, 5):
        raise ValueError("Stage38 freezes G=8, active width=8, elite width=5")
    expected_elite = (batch, groups, elite_width, steps)
    if predictions["stage38_public_elite_bc_log_probs"].shape != expected_elite:
        raise ValueError("Stage38 public elite BC replay shape drifted")
    scalar_expectations = {
        "diffgrpo_group_size": 8,
        "stage38_sampled_chain_count": 320,
        "stage38_counterfactual_selector_count": 224,
        "stage38_current_physical_replay_count": 64,
        "stage38_public_physical_replay_count": 40,
        "stage38_reward_dependent_replay": 1,
        "stage38_replay_deduplicated": 1,
    }
    for name, expected in scalar_expectations.items():
        actual = int(predictions[name].detach().item())
        if actual != expected:
            raise ValueError(
                f"Stage38 trace accounting drifted: {name}={actual}"
            )
    current_unique = int(
        predictions["stage38_current_unique_replay_count"].detach().item()
    )
    public_unique = int(
        predictions["stage38_public_unique_replay_count"].detach().item()
    )
    if not 0 < current_unique <= 64 or not 0 < public_unique <= 40:
        raise ValueError("Stage38 unique replay accounting is invalid")

    modes = 20
    rewards = _bank(
        predictions["raw_rewards"], batch=batch, groups=groups, modes=modes
    )
    valid = _bank(
        predictions["reward_valid_mask"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    components = _bank(
        predictions["component_scores"],
        batch=batch,
        groups=groups,
        modes=modes,
        trailing=(6,),
    )
    base_rewards = _bank(
        predictions["diffgrpo_base_rewards"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    base_valid = _bank(
        predictions["diffgrpo_base_valid_mask"],
        batch=batch,
        groups=groups,
        modes=modes,
    )
    base_components = _bank(
        predictions["diffgrpo_base_component_scores"],
        batch=batch,
        groups=groups,
        modes=modes,
        trailing=(6,),
    )
    objective = compute_elite_set_repair_objective(
        current_log_probs=current_log_probs,
        reference_mean_kl=predictions["diffgrpo_reference_mean_kl"],
        public_elite_bc_log_probs=predictions[
            "stage38_public_elite_bc_log_probs"
        ],
        rewards=rewards,
        valid_mask=valid,
        component_scores=components,
        base_rewards=base_rewards,
        base_valid_mask=base_valid,
        base_component_scores=base_components,
        current_selected_modes=predictions[
            "stage38_current_selected_modes"
        ],
        scene_buckets=predictions["stage38_scene_buckets"],
        active_modes=predictions["stage38_active_modes"],
        active_valid=predictions["stage38_active_valid"],
        public_elite_modes=predictions["stage38_public_elite_modes"],
        public_elite_valid=predictions["stage38_public_elite_valid"],
        elite_width=elite_width,
        positive_challenger_width=int(getattr(
            config, "stage38_positive_challenger_width", 2
        )),
        active_pool_width=active_width,
        positive_margin=float(getattr(
            config, "stage38_positive_margin", 0.001
        )),
        negative_tolerance=float(getattr(
            config, "stage38_negative_tolerance", 0.0001
        )),
        advantage_scale=float(getattr(
            config, "stage38_advantage_scale", 0.002
        )),
        advantage_clip=float(getattr(
            config, "stage38_advantage_clip", 2.0
        )),
        mature_positive_multiplier=float(getattr(
            config, "stage38_mature_positive_multiplier", 0.25
        )),
        safety_tolerance=float(getattr(
            config, "diffgrpo_safety_regression_tolerance", 1e-6
        )),
        step_discount=float(getattr(config, "stage38_step_discount", 0.6)),
        bc_weight=float(getattr(config, "stage38_bc_weight", 0.1)),
        kl_weight=float(getattr(config, "stage38_kl_weight", 0.1)),
        safety_kl_weight=float(getattr(
            config, "stage38_safety_kl_weight", 0.5
        )),
    )
    exact_targets = (
        (
            predictions["stage38_delta_set"].float(),
            objective["_target_delta_set"].float(),
        ),
        (predictions["stage38_escr_advantages"].float(),
         objective["_target_advantages"].float()),
        (predictions["stage38_union_safe_oracle_gain"].float(),
         objective["_target_union_safe_oracle_gain"].float()),
    )
    if not all(torch.equal(actual, expected) for actual, expected in exact_targets):
        raise RuntimeError("Stage38 replay targets drifted after reward freeze")

    reward_mask = valid.bool() & torch.isfinite(rewards)
    valid_rewards = rewards[reward_mask].float()
    zero = objective["loss"].detach() * 0.0
    result = {
        "loss": objective["loss"],
        "generation_grpo_loss": objective["policy_loss"],
        "diffgrpo_bc_loss": objective["elite_bc_loss"],
        "generation_reference_kl_loss": objective["active_kl_loss"],
        "generation_kl_loss": objective["active_kl_loss"],
        "diffgrpo_mean_current_log_prob": objective[
            "mean_current_log_prob"
        ],
        "raw_reward_mean": (
            valid_rewards.mean().detach() if valid_rewards.numel() else zero
        ),
        "raw_reward_std": (
            valid_rewards.std(unbiased=False).detach()
            if valid_rewards.numel() else zero
        ),
    }
    for name in (
        "policy_loss",
        "elite_bc_loss",
        "active_kl_loss",
        "delta_set_mean",
        "positive_fraction",
        "negative_fraction",
        "positive_challenger_fraction",
        "negative_elite_fraction",
        "invalid_elite_fraction",
        "safety_veto_fraction",
        "active_fraction",
        "union_safe_oracle_gain_mean",
        "selected_set_gain_mean",
        "public_elite_current_delta_mean",
        "hard_union_safe_oracle_gain_mean",
        "mature_union_safe_oracle_gain_mean",
        "public_elite_exact_width",
        "active_pool_bounded",
        "advantage_abs_mean",
        "mean_current_log_prob",
        "mean_exact_kl",
    ):
        result[f"stage38_{name}"] = objective[name]
    for name in (
        "stage38_sampled_chain_count",
        "stage38_counterfactual_selector_count",
        "stage38_current_unique_replay_count",
        "stage38_public_unique_replay_count",
        "stage38_current_physical_replay_count",
        "stage38_public_physical_replay_count",
        "stage38_reward_dependent_replay",
        "stage38_replay_deduplicated",
        "stage38_selector_mode_disagreement",
        "stage38_current_selector_switch_rate",
        "stage38_public_selector_switch_rate",
    ):
        result[name] = predictions[name]
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Stage38 loss bridge produced non-finite output")
    return result
