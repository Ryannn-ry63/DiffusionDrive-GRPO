"""Frozen Stage37 configuration contract shared by launch and validation."""

from navsim.agents.diffusiondrive.stage37_bistate_projected import (
    STAGE37_OBJECTIVE_REVISION,
    STAGE37_PLAN_SHA256,
)


STAGE37_ALLOWED_INITIAL_SHA256 = {
    # Released public 88.1 checkpoint; Stage37 may not inherit Stage35/36.
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b",
}

STAGE37_LEGACY_STAGE31_KEYS = (
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

STAGE37_ROUTE_EXPECTED = {
    "generation_policy_algorithm": "diffgrpo_bistate_projected_deployment",
    "diffgrpo_group_size": 8,
    "stage37_plan_sha256": STAGE37_PLAN_SHA256,
    "stage37_objective_revision": STAGE37_OBJECTIVE_REVISION,
    "stage37_active_pool_width": 4,
    "stage37_selector_micro_batch_size": 16,
    "stage37_frontier_width": 5,
    "stage37_tail_elite_count": 2,
    "stage37_frontier_regression_tolerance": 1e-4,
    "stage37_tail_margin": 0.001,
    "stage37_advantage_scale": 0.002,
    "stage37_advantage_clip": 2.0,
    "stage37_counterfactual_weight": 0.25,
    "stage37_tail_weight": 0.25,
    "stage37_frontier_weight": 0.25,
    "stage37_projection_recovery_coefficient": 0.25,
    "stage37_projection_epsilon": 1e-12,
    "stage37_bc_weight": 0.1,
    "stage37_kl_weight": 0.1,
    "stage37_safety_kl_weight": 0.5,
    "stage37_optimizer_steps_per_epoch": 48,
    "stage37_gradient_accumulation": 8,
    "stage37_global_bucket_composition": (2, 30, 8, 24),
}
