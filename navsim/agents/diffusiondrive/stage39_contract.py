"""Fail-closed Stage39 route contract."""

from navsim.agents.diffusiondrive.stage39_challenger import (
    STAGE39_OBJECTIVE_REVISION,
    STAGE39_PLAN_SHA256,
)


STAGE39_PUBLIC_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
STAGE39_TRAINING_MODES = {
    "stage39_challenger_bc",
    "stage39_challenger_standard_grpo",
    "stage39_challenger_set_grpo",
}
STAGE39_BRANCH_BY_MODE = {
    "stage39_challenger_bc": "BC",
    "stage39_challenger_standard_grpo": "STD",
    "stage39_challenger_set_grpo": "SET",
}
STAGE39_ROUTE_EXPECTED = {
    "generation_policy_algorithm": "diffgrpo_non_destructive_challenger",
    "grpo_decoder_gradient_scope": "all_layers",
    "grpo_decoder_layer0_lr_mult": 0.1,
    "stage39_plan_sha256": STAGE39_PLAN_SHA256,
    "stage39_objective_revision": STAGE39_OBJECTIVE_REVISION,
    "stage39_public_candidate_count": 20,
    "stage39_challenger_candidate_count": 20,
    "stage39_group_size": 8,
    "stage39_elite_width": 5,
    "stage39_positive_margin": 0.001,
    "stage39_advantage_scale": 0.02,
    "stage39_advantage_clip": 2.0,
    "stage39_std_floor": 0.0001,
    "stage39_kl_weight": 0.1,
    "stage39_step_discount": 0.6,
    "stage39_optimizer_steps_per_epoch": 48,
    "stage39_gradient_accumulation": 8,
    "stage39_global_bucket_composition": (2, 30, 8, 24),
    "diffusion_truncation_timestep": 32,
    "diffusion_roll_timesteps": (32, 24, 16, 8, 0),
    "diffusion_scheduler_num_inference_steps": 125,
    "weight_decay": 0.0,
}

