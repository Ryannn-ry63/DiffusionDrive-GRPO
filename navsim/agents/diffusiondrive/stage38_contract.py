"""Frozen Stage38 ESCR configuration contract."""

from navsim.agents.diffusiondrive.stage38_elite_set_repair import (
    STAGE38_OBJECTIVE_REVISION,
    STAGE38_PLAN_SHA256,
)


STAGE38_ALLOWED_INITIAL_SHA256 = {
    # Fold-matched Stage36 RGT192 checkpoints.
    "2a2302da415e38aa654c172858d3f89f8112f774a6215d89b9c150112f0dab33",
    "0bafe817ebfe98032b3d4e9499eb78d547efcf7230ddc3c83aa4b1bd4e4e7a9b",
}

STAGE38_LEGACY_STAGE31_KEYS = (
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

STAGE38_ROUTE_EXPECTED = {
    "generation_policy_algorithm": "diffgrpo_elite_set_counterfactual_repair",
    "generation_advantage_mode": "escr_set_marginal",
    "diffgrpo_group_size": 8,
    "stage38_plan_sha256": STAGE38_PLAN_SHA256,
    "stage38_objective_revision": STAGE38_OBJECTIVE_REVISION,
    "stage38_selector_micro_batch_size": 16,
    "stage38_elite_width": 5,
    "stage38_positive_challenger_width": 2,
    "stage38_active_pool_width": 8,
    "stage38_positive_margin": 0.001,
    "stage38_negative_tolerance": 0.0001,
    "stage38_advantage_scale": 0.002,
    "stage38_advantage_clip": 2.0,
    "stage38_mature_positive_multiplier": 0.25,
    "stage38_bc_weight": 0.1,
    "stage38_kl_weight": 0.1,
    "stage38_safety_kl_weight": 0.5,
    "stage38_step_discount": 0.6,
    "stage38_optimizer_steps_per_epoch": 48,
    "stage38_gradient_accumulation": 8,
    "stage38_global_bucket_composition": (2, 30, 8, 24),
}
