"""Frozen Stage34 configuration contract shared by launch and validation code."""

STAGE34_PLAN_SHA256 = (
    "ad3e9dace53cd3196ef236c7d3630605e64aa46c908e5dd18c81b8600eaf155a"
)
STAGE34_OBJECTIVE_REVISION = "mode_aligned_frontier_v1"

STAGE34_LEGACY_STAGE30_KEYS = (
    "stage30_top_k",
    "stage30_delta_scale_floor",
    "stage30_coverage_top_weight",
    "stage30_boundary_top_weight",
    "stage30_boundary_positive_margin",
    "stage30_boundary_negative_multiplier",
    "stage30_mature_negative_floor",
    "stage30_mature_positive_margin",
    "stage30_mature_negative_multiplier",
    "stage30_component_tolerance",
    "stage30_plan_sha256",
    "stage30_optimizer_steps_per_epoch",
    "stage30_gradient_accumulation",
    "stage30_global_bucket_composition",
)

STAGE34_ROUTE_EXPECTED = {
    "generation_policy_algorithm": "diffgrpo_mode_aligned_frontier",
    "diffgrpo_group_size": 20,
    "stage34_plan_sha256": STAGE34_PLAN_SHA256,
    "stage34_objective_revision": STAGE34_OBJECTIVE_REVISION,
    "stage34_top_k": 5,
    "stage34_headroom_k": 2,
    "stage34_delta_scale_floor": 0.002,
    "stage34_deployment_weight": 0.5,
    "stage34_headroom_weight": 0.25,
    "stage34_headroom_margin": 0.001,
    "stage34_top5_negative_multiplier": 1.5,
    "stage34_top1_negative_multiplier": 2.0,
    "stage34_mature_positive_multiplier": 0.25,
    "stage34_advantage_clip": 2.0,
    "stage34_bc_weight": 0.1,
    "stage34_kl_weight": 0.1,
    "stage34_safety_kl_weight": 0.5,
    "stage34_optimizer_steps_per_epoch": 48,
    "stage34_gradient_accumulation": 8,
    "stage34_global_bucket_composition": (2, 30, 8, 24),
}
