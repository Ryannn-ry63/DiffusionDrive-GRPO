"""Frozen Stage36 configuration contract shared by launch and validation."""

from navsim.agents.diffusiondrive.stage36_reference_gated_tail import (
    STAGE36_OBJECTIVE_REVISION,
    STAGE36_PLAN_SHA256,
)


STAGE36_ALLOWED_INITIAL_SHA256 = {
    # Fold-matched Stage32 DPEL192 initializers.
    "2c01b93fb35b246b6c897db149f38c13cc4a4259933f8ef21aee4d311dfee951",
    "b2bf4243e622235de4003938fba85240d2fd660ad0a7123b750cbc83abf6a557",
}

STAGE36_LEGACY_STAGE31_KEYS = (
    "stage31_plan_sha256",
    "stage31_headroom_low",
    "stage31_headroom_high",
    "stage31_delta_scale_floor",
    "stage31_rank_weight",
    "stage31_bc_weight",
    "stage31_kl_weight",
    "stage31_safety_kl_weight",
    "stage31_optimizer_steps_per_epoch",
    "stage31_gradient_accumulation",
    "stage31_global_bucket_composition",
)

STAGE36_ROUTE_EXPECTED = {
    "generation_policy_algorithm": "diffgrpo_reference_gated_tail_ncd",
    "diffgrpo_group_size": 8,
    "stage36_plan_sha256": STAGE36_PLAN_SHA256,
    "stage36_objective_revision": STAGE36_OBJECTIVE_REVISION,
    "stage36_active_pool_width": 4,
    "stage36_selector_micro_batch_size": 16,
    "stage36_frontier_risk_margin": 0.10,
    "stage36_frontier_width": 5,
    "stage36_tail_elite_count": 2,
    "stage36_retention_tolerance": 1e-4,
    "stage36_tail_margin": 0.001,
    "stage36_tail_scale": 0.002,
    "stage36_tail_weight": 0.25,
    "stage36_retention_weight": 0.25,
    "stage36_counterfactual_weight": 0.25,
    "stage36_mature_positive_multiplier": 0.25,
    "stage36_advantage_eps": 1e-3,
    "stage36_delta_scale_floor": 0.002,
    "stage36_headroom_low": 0.75,
    "stage36_headroom_high": 0.90,
    "stage36_rank_weight": 0.5,
    "stage36_advantage_clip": 2.0,
    "stage36_bc_weight": 0.1,
    "stage36_kl_weight": 0.1,
    "stage36_safety_kl_weight": 0.5,
    "stage36_optimizer_steps_per_epoch": 48,
    "stage36_gradient_accumulation": 8,
    "stage36_global_bucket_composition": (2, 30, 8, 24),
    "stage36_signal_retention_fraction_min": 0.01,
    "stage36_signal_tail_fraction_min": 0.005,
}

