"""Loss routing and accounting checks for Stage39 challenger training."""

from __future__ import annotations

from typing import Any, Dict

import torch

from navsim.agents.diffusiondrive.stage39_challenger import (
    compute_stage39_challenger_objective,
)
from navsim.agents.diffusiondrive.stage39_contract import (
    STAGE39_BRANCH_BY_MODE,
)


def _bank(
    value: torch.Tensor,
    *,
    batch: int,
    groups: int = 8,
    modes: int = 20,
    trailing: tuple[int, ...] = (),
) -> torch.Tensor:
    expected = (batch, groups * modes, *trailing)
    if value.shape != expected:
        raise ValueError(
            f"Stage39 bank must have shape {expected}, got {tuple(value.shape)}"
        )
    return value.reshape(batch, groups, modes, *trailing)


def compute_stage39_transfuser_loss(
    predictions: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    required = (
        "stage39_challenger_log_probs",
        "stage39_challenger_reference_kl",
        "stage39_public_bc_log_probs",
        "raw_rewards",
        "reward_valid_mask",
        "component_scores",
        "diffgrpo_base_rewards",
        "diffgrpo_base_valid_mask",
        "diffgrpo_base_component_scores",
        "stage39_sampled_chain_count",
        "stage39_replayed_chain_count",
        "stage39_independent_initial_noise",
        "stage39_initial_noise_abs_cosine",
        "diffgrpo_group_size",
    )
    missing = [name for name in required if predictions.get(name) is None]
    if missing:
        raise ValueError(f"Missing Stage39 tensors: {missing}")
    log_probs = predictions["stage39_challenger_log_probs"]
    if log_probs.ndim != 4 or log_probs.shape[1:3] != (8, 20):
        raise ValueError("Stage39 log probabilities must be [B,8,20,S]")
    if int(predictions["diffgrpo_group_size"].item()) != 8:
        raise ValueError("Stage39 group size drifted")
    for name, expected in (
        ("stage39_sampled_chain_count", 320),
        ("stage39_replayed_chain_count", 320),
        ("stage39_independent_initial_noise", 1),
    ):
        if int(predictions[name].detach().item()) != expected:
            raise ValueError(f"Stage39 accounting drifted: {name}")
    batch = log_probs.shape[0]
    branch = STAGE39_BRANCH_BY_MODE.get(
        str(getattr(config, "grpo_training_mode", ""))
    )
    if branch is None:
        raise ValueError("Stage39 loss received an unregistered training mode")
    objective = compute_stage39_challenger_objective(
        branch=branch,
        challenger_log_probs=log_probs,
        challenger_reference_kl=predictions[
            "stage39_challenger_reference_kl"
        ],
        public_bc_log_probs=predictions["stage39_public_bc_log_probs"],
        challenger_rewards=_bank(predictions["raw_rewards"], batch=batch),
        challenger_valid_mask=_bank(
            predictions["reward_valid_mask"], batch=batch
        ),
        challenger_component_scores=_bank(
            predictions["component_scores"], batch=batch, trailing=(6,)
        ),
        public_rewards=_bank(
            predictions["diffgrpo_base_rewards"], batch=batch
        ),
        public_valid_mask=_bank(
            predictions["diffgrpo_base_valid_mask"], batch=batch
        ),
        public_component_scores=_bank(
            predictions["diffgrpo_base_component_scores"],
            batch=batch,
            trailing=(6,),
        ),
        elite_width=int(getattr(config, "stage39_elite_width", 5)),
        positive_margin=float(getattr(
            config, "stage39_positive_margin", 0.001
        )),
        advantage_scale=float(getattr(
            config, "stage39_advantage_scale", 0.02
        )),
        advantage_clip=float(getattr(
            config, "stage39_advantage_clip", 2.0
        )),
        standard_deviation_floor=float(getattr(
            config, "stage39_std_floor", 1e-4
        )),
        safety_tolerance=float(getattr(
            config, "diffgrpo_safety_regression_tolerance", 1e-6
        )),
        step_discount=float(getattr(
            config, "stage39_step_discount", 0.6
        )),
        kl_weight=float(getattr(config, "stage39_kl_weight", 0.1)),
    )
    result = {
        "loss": objective["loss"],
        "generation_grpo_loss": objective["policy_loss"],
        "diffgrpo_bc_loss": objective["bc_loss"],
        "generation_reference_kl_loss": objective["kl_loss"],
        "generation_kl_loss": objective["kl_loss"],
        "stage39_branch_id": log_probs.new_tensor(
            float(("BC", "STD", "SET").index(branch))
        ),
    }
    for name in (
        "policy_loss",
        "bc_loss",
        "kl_loss",
        "advantage_abs_mean",
        "standard_positive_fraction",
        "set_positive_fraction",
        "safety_veto_fraction",
        "set_delta_mean",
        "union_safe_oracle_gain_mean",
        "public_group_valid_fraction",
    ):
        result[f"stage39_{name}"] = objective[name]
    for name in (
        "stage39_sampled_chain_count",
        "stage39_replayed_chain_count",
        "stage39_independent_initial_noise",
        "stage39_initial_noise_abs_cosine",
    ):
        result[name] = predictions[name]
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Stage39 loss bridge produced non-finite values")
    return result

