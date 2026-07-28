"""Frozen Stage35 configuration contract shared by launch and validation."""

from navsim.agents.diffusiondrive.stage35_counterfactual import (
    STAGE35_OBJECTIVE_REVISION,
    STAGE35_PLAN_SHA256,
)


STAGE35_ALLOWED_INITIAL_SHA256 = {
    # Stage32 DPEL192, fold-matched pilots selected before Stage35.
    "2c01b93fb35b246b6c897db149f38c13cc4a4259933f8ef21aee4d311dfee951",
    "b2bf4243e622235de4003938fba85240d2fd660ad0a7123b750cbc83abf6a557",
}


STAGE35_LEGACY_STAGE31_KEYS = (
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

STAGE35_ROUTE_EXPECTED = {
    "generation_policy_algorithm": (
        "diffgrpo_nested_counterfactual_deployment"
    ),
    "diffgrpo_group_size": 8,
    "stage35_plan_sha256": STAGE35_PLAN_SHA256,
    "stage35_objective_revision": STAGE35_OBJECTIVE_REVISION,
    "stage35_active_pool_width": 4,
    "stage35_selector_micro_batch_size": 16,
    "stage35_frontier_risk_margin": 0.10,
    "stage35_counterfactual_weight": 0.25,
    "stage35_mature_positive_multiplier": 0.25,
    "stage35_advantage_eps": 1e-3,
    "stage35_delta_scale_floor": 0.002,
    "stage35_headroom_low": 0.75,
    "stage35_headroom_high": 0.90,
    "stage35_rank_weight": 0.5,
    "stage35_advantage_clip": 2.0,
    "stage35_bc_weight": 0.1,
    "stage35_kl_weight": 0.1,
    "stage35_safety_kl_weight": 0.5,
    "stage35_optimizer_steps_per_epoch": 48,
    "stage35_gradient_accumulation": 8,
    "stage35_global_bucket_composition": (2, 30, 8, 24),
    "stage35_signal_nonzero_fraction_min": 0.10,
    "stage35_signal_nonowner_scene_fraction_min": 0.25,
}
