from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/pytorch_model.bin"
    #plan_anchor_path: str = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/anchor_64_kmeans.npy"
    plan_anchor_path: str = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/kmeans_navsim_traj_20.npy"
    
    #metric_cache_path: str = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache"
    metric_cache_path: str = "/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache_trainval"
    # GRPO needs an online PDM reward for every sampled scene. Restrict cached
    # training/validation datasets to tokens present in metric_cache_path.
    filter_training_by_metric_cache: bool = True
    # Lazy-load metric caches on first use (fast startup). 0 = unlimited entries in RAM; set e.g. 16384 to cap memory.
    metric_cache_lru_max: int = 0
    # Rewards depend on current trajectories and must not be reused across batches.
    reward_compute_interval: int = 1
    
    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    policy_loss_weight: float = 1.0
    kl_loss_weight: float = 0.01
    grpo_clip_ratio: float = 0.2
    grpo_advantage_eps: float = 1e-3
    grpo_old_policy_sync_steps: int = 32
    # Set to -1 for gated ablations that must retain every epoch checkpoint.
    grpo_checkpoint_save_top_k: int = 2
    grpo_checkpoint_every_n_train_steps: int = 0
    grpo_training_mode: str = "classification_shared"
    grpo_decoder_gradient_scope: str = "last_layer"
    # Deployment-only selector. Reference logits choose among current generator
    # candidates and never participate in the training objective. value_top2
    # conservatively reranks only the frozen-reference selector's first two modes.
    inference_selector_source: str = "current"
    value_selector_num_heads: int = 3
    value_selector_top_k: int = 2
    value_selector_pair_reward_gap: float = 0.01
    value_selector_bootstrap_fraction: float = 0.8
    value_selector_checkpoint_path: str = ""
    value_selector_train_manifest_path: str = ""
    value_selector_calibration_margin: float = -1.0
    value_selector_safety_threshold: float = 0.9
    value_selector_confidence_z: float = 1.64
    stage23_selector_num_members: int = 5
    stage23_selector_num_modes: int = 20
    stage23_selector_dim: int = 128
    stage23_selector_checkpoint_path: str = ""
    stage23_selector_calibration_path: str = ""
    stage23_selector_train_manifest_path: str = ""
    stage23_candidate_bank_paths: Tuple[str, ...] = ()
    stage23_generator_train_manifest_path: str = ""
    stage23_selector_pair_reward_gap: float = 0.01
    stage23_selector_focal_gamma: float = 2.0
    stage23_selector_unsafe_positive_weight: float = 50.0
    stage23_selector_residual_margin: float = -1.0
    stage23_selector_safety_threshold: float = 0.95
    stage23_selector_confidence_z: float = 1.96
    stage23_selector_safety_tolerance: float = 0.0
    # Stage24 cross-generator safety/value selector.
    stage24_selector_num_members: int = 8
    stage24_selector_num_modes: int = 20
    stage24_selector_dim: int = 128
    stage24_selector_checkpoint_path: str = ""
    stage24_selector_calibration_path: str = ""
    stage24_selector_train_manifest_path: str = ""
    stage24_candidate_bank_paths: Tuple[str, ...] = ()
    stage24_selector_steps_per_epoch: int = 1528
    stage24_selector_subbag_fraction: float = 0.8
    stage24_selector_pair_reward_gap: float = 0.005
    stage24_selector_focal_gamma: float = 2.0
    stage24_selector_hard_negative_count: int = 4
    stage24_selector_residual_margin: float = -1.0
    stage24_selector_risk_threshold: float = -1.0
    stage24_selector_ood_threshold: float = -1.0
    stage24_selector_confidence_z: float = 1.96
    stage24_selector_safety_tolerance: float = 0.001
    stage24_collect_calibration: bool = False
    stage24_generator_train_manifest_path: str = ""
    # Stage25 relative-to-fallback harm gate over frozen Stage24 embeddings.
    stage25_selector_checkpoint_path: str = ""
    stage25_selector_calibration_path: str = ""
    stage25_selector_focal_gamma: float = 2.0
    stage25_selector_hard_negative_count: int = 4
    stage25_selector_hard_risk_margin: float = 0.9
    stage25_selector_risk_threshold: float = -1.0
    stage25_collect_calibration: bool = False
    # Stage28 may use an unsafe-but-informative selector only for guarded
    # training exploration. Evaluation always uses the safe multi selector.
    stage28_training_selector_role: str = ""
    stage28_exploration_authorization_path: str = ""
    paired_risk_checkpoint_path: str = ""
    paired_risk_train_manifest_path: str = ""
    paired_risk_validation_manifest_path: str = ""
    paired_risk_calibration_path: str = ""
    paired_risk_threshold: float = -1.0
    paired_risk_positive_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    paired_risk_selected_mode_weight: float = 4.0
    paired_risk_delta_loss_weight: float = 0.25
    paired_risk_expected_generator_sha256: str = ""
    paired_risk_expected_reference_sha256: str = ""
    selection_temperature: float = 1.0
    selection_entropy_weight: float = 0.0
    selection_exploration_floor: float = 0.0
    selection_behavior_weighting: str = "old_policy"
    selection_rank_loss_weight: float = 0.0
    selection_rank_reward_gap: float = 0.01
    selection_rank_reward_scale: float = 0.10
    selection_rank_logit_margin: float = 0.20
    grpo_scene_weight_mode: str = "uniform"
    grpo_reference_gate_margin: float = 0.01
    grpo_reference_gate_scale: float = 0.05
    # Generation-only GRPO may change the shared features consumed by the
    # frozen selector. This optional KL preserves the fixed-reference
    # selector distribution without adding a selection policy gradient.
    selector_consistency_kl_weight: float = 0.0
    grpo_reward_mode: str = "pdms"
    # Selector GRPO updates the shared/regression decoder too, so constrain
    # both denoising means to the independent frozen base decoder.
    selector_generation_kl_weight: float = 0.0

    pdm_tiebreak_max_epsilon: float = 1e-3
    pdm_dense_weight: float = 0.1
    generation_policy_loss_weight: float = 1.0
    generation_kl_loss_weight: float = 0.1
    generation_policy_algorithm: str = "legacy_ppo"
    diffgrpo_bc_weight: float = 0.1
    diffgrpo_step_discount: float = 0.6
    diffgrpo_logprob_reduction: str = "mean"
    diffgrpo_train_manifest_path: str = ""
    diffgrpo_selected_mode_manifest_path: str = ""
    diffgrpo_group_size: int = 8
    diffgrpo_base_margin: float = 0.01
    diffgrpo_base_scale: float = 0.10
    diffgrpo_base_advantage_clip: float = 2.0
    diffgrpo_safety_regression_tolerance: float = 1e-6
    diffgrpo_bc_max_weight: float = 0.3
    diffgrpo_paired_positive_margin: float = 0.01
    diffgrpo_paired_negative_margin: float = 0.01
    diffgrpo_paired_mature_negative_margin: float = 0.002
    diffgrpo_paired_mature_reward_threshold: float = 0.75
    diffgrpo_paired_mature_negative_multiplier: float = 2.0
    diffgrpo_paired_regular_kl_weight: float = 0.1
    diffgrpo_paired_mature_kl_weight: float = 0.5
    diffgrpo_paired_bootstrap_advantage_weight: float = 0.25
    # Stage29 reference-anchored headroom GRPO.
    stage29_headroom_low: float = 0.75
    stage29_headroom_high: float = 0.90
    stage29_delta_scale_floor: float = 0.002
    stage29_rank_weight: float = 0.5
    stage29_conditional_regularization: bool = False
    stage29_fixed_bc_weight: float = 0.1
    stage29_fixed_kl_weight: float = 0.1
    stage29_hard_bc_weight: float = 0.05
    stage29_mature_bc_weight: float = 0.1
    stage29_hard_kl_weight: float = 0.05
    stage29_mature_kl_weight: float = 0.5
    stage29_safety_kl_weight: float = 0.5
    # Stage30 Mode-Coverage Constrained GRPO (frozen 2026-07-25 plan).
    stage30_bucket_manifest_path: str = ""
    stage30_plan_sha256: str = (
        "e9867ec48d4806ff97adb0284bae7550305cb1350e4ec66dbfd20d245de435ce"
    )
    stage30_top_k: int = 5
    stage30_delta_scale_floor: float = 0.002
    stage30_coverage_top_weight: float = 0.5
    stage30_boundary_top_weight: float = 0.25
    stage30_boundary_positive_margin: float = 0.001
    stage30_boundary_negative_multiplier: float = 2.0
    stage30_mature_negative_floor: float = -0.0002
    stage30_mature_positive_margin: float = 0.002
    stage30_mature_negative_multiplier: float = 4.0
    stage30_component_tolerance: float = 1e-6
    stage30_optimizer_steps_per_epoch: int = 48
    stage30_gradient_accumulation: int = 8
    stage30_global_bucket_composition: tuple = (2, 30, 8, 24)
    # Stage31 selector-consistent decision-level GRPO (frozen 2026-07-25).
    stage31_bucket_manifest_path: str = ""
    stage31_plan_sha256: str = (
        "3ff3d3bcd3120b9d73a017876f942458eb3fa16651517e4e30b27cf2fff7bb0d"
    )
    stage31_headroom_low: float = 0.75
    stage31_headroom_high: float = 0.90
    stage31_delta_scale_floor: float = 0.002
    stage31_rank_weight: float = 0.5
    stage31_bc_weight: float = 0.1
    stage31_kl_weight: float = 0.1
    stage31_safety_kl_weight: float = 0.5
    stage31_optimizer_steps_per_epoch: int = 48
    stage31_gradient_accumulation: int = 8
    stage31_global_bucket_composition: tuple = (2, 30, 8, 24)
    # Stage32 selector-aware generator frontier credit (frozen 2026-07-26).
    stage32_bucket_manifest_path: str = ""
    stage32_plan_sha256: str = ""
    stage32_scf_objective_revision: str = "mean_normalized_bc_kl_v2"
    stage32_frontier_pool_size: int = 4
    stage32_frontier_risk_margin: float = 0.10
    stage32_frontier_owner_margin: float = 0.001
    stage32_frontier_weight: float = 0.5
    stage32_frontier_mature_cap: float = 0.25
    stage32_optimizer_steps_per_epoch: int = 48
    stage32_gradient_accumulation: int = 8
    stage32_global_bucket_composition: tuple = (2, 30, 8, 24)
    # Stage33 counterfactual deployment-credit GRPO (frozen 2026-07-26).
    stage33_bucket_manifest_path: str = ""
    stage33_plan_sha256: str = ""
    stage33_selector_reranker_path: str = ""
    stage33_selector_reranker_sha256: str = ""
    stage33_cdc_objective_revision: str = "cdc_deployment_credit_v1"
    stage33_deployment_weight: float = 0.5
    stage33_headroom_weight: float = 0.5
    stage33_headroom_margin: float = 0.001
    stage33_advantage_clip: float = 2.0
    stage33_optimizer_steps_per_epoch: int = 48
    stage33_gradient_accumulation: int = 8
    stage33_global_bucket_composition: tuple = (2, 30, 8, 24)
    # Stage34 mode-aligned deployable-frontier GRPO (frozen 2026-07-27).
    stage34_bucket_manifest_path: str = ""
    stage34_plan_sha256: str = (
        "ad3e9dace53cd3196ef236c7d3630605e64aa46c908e5dd18c81b8600eaf155a"
    )
    stage34_objective_revision: str = "mode_aligned_frontier_v1"
    stage34_top_k: int = 5
    stage34_headroom_k: int = 2
    stage34_delta_scale_floor: float = 0.002
    stage34_deployment_weight: float = 0.5
    stage34_headroom_weight: float = 0.25
    stage34_headroom_margin: float = 0.001
    stage34_top5_negative_multiplier: float = 1.5
    stage34_top1_negative_multiplier: float = 2.0
    stage34_mature_positive_multiplier: float = 0.25
    stage34_advantage_clip: float = 2.0
    stage34_bc_weight: float = 0.1
    stage34_kl_weight: float = 0.1
    stage34_safety_kl_weight: float = 0.5
    stage34_optimizer_steps_per_epoch: int = 48
    stage34_gradient_accumulation: int = 8
    stage34_global_bucket_composition: tuple = (2, 30, 8, 24)
    # Stage35 nested counterfactual deployment GRPO (frozen 2026-07-27).
    stage35_bucket_manifest_path: str = ""
    stage35_plan_sha256: str = (
        "301e5fb37066c34f6a3e224be08fd1ca435cc9a8d961121869cf6dcd21d2fae7"
    )
    stage35_objective_revision: str = "nested_counterfactual_deployment_v1"
    stage35_active_pool_width: int = 4
    stage35_selector_micro_batch_size: int = 16
    stage35_frontier_risk_margin: float = 0.10
    stage35_counterfactual_weight: float = 0.25
    stage35_mature_positive_multiplier: float = 0.25
    stage35_advantage_eps: float = 1e-3
    stage35_delta_scale_floor: float = 0.002
    stage35_headroom_low: float = 0.75
    stage35_headroom_high: float = 0.90
    stage35_rank_weight: float = 0.5
    stage35_advantage_clip: float = 2.0
    stage35_bc_weight: float = 0.1
    stage35_kl_weight: float = 0.1
    stage35_safety_kl_weight: float = 0.5
    stage35_optimizer_steps_per_epoch: int = 48
    stage35_gradient_accumulation: int = 8
    stage35_global_bucket_composition: tuple = (2, 30, 8, 24)
    stage35_signal_nonzero_fraction_min: float = 0.10
    stage35_signal_nonowner_scene_fraction_min: float = 0.25
    stage35_diagnostic_only: bool = False
    # Stage36 reference-gated tail NCD-GRPO (frozen 2026-07-28).
    stage36_bucket_manifest_path: str = ""
    stage36_plan_sha256: str = (
        "858e6688faa9dff6d9027254e6a0262c67e47582d51a5829cc12a9723a25e521"
    )
    stage36_objective_revision: str = "reference_gated_tail_ncd_v1"
    stage36_active_pool_width: int = 4
    stage36_selector_micro_batch_size: int = 16
    stage36_frontier_risk_margin: float = 0.10
    stage36_frontier_width: int = 5
    stage36_tail_elite_count: int = 2
    stage36_retention_tolerance: float = 1e-4
    stage36_tail_margin: float = 0.001
    stage36_tail_scale: float = 0.002
    stage36_tail_weight: float = 0.25
    stage36_retention_weight: float = 0.25
    stage36_counterfactual_weight: float = 0.25
    stage36_mature_positive_multiplier: float = 0.25
    stage36_advantage_eps: float = 1e-3
    stage36_delta_scale_floor: float = 0.002
    stage36_headroom_low: float = 0.75
    stage36_headroom_high: float = 0.90
    stage36_rank_weight: float = 0.5
    stage36_advantage_clip: float = 2.0
    stage36_bc_weight: float = 0.1
    stage36_kl_weight: float = 0.1
    stage36_safety_kl_weight: float = 0.5
    stage36_optimizer_steps_per_epoch: int = 48
    stage36_gradient_accumulation: int = 8
    stage36_global_bucket_composition: tuple = (2, 30, 8, 24)
    stage36_signal_retention_fraction_min: float = 0.01
    stage36_signal_tail_fraction_min: float = 0.005
    stage36_diagnostic_only: bool = False
    # Stage37 bi-state projected deployment GRPO and JFI selector.
    stage37_bucket_manifest_path: str = ""
    stage37_plan_sha256: str = (
        "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
    )
    stage37_objective_revision: str = "bistate_projected_deployment_v1"
    stage37_active_pool_width: int = 4
    stage37_selector_micro_batch_size: int = 16
    stage37_frontier_width: int = 5
    stage37_tail_elite_count: int = 2
    stage37_frontier_regression_tolerance: float = 1e-4
    stage37_tail_margin: float = 0.001
    stage37_advantage_scale: float = 0.002
    stage37_advantage_clip: float = 2.0
    stage37_counterfactual_weight: float = 0.25
    stage37_tail_weight: float = 0.25
    stage37_frontier_weight: float = 0.25
    stage37_projection_recovery_coefficient: float = 0.25
    stage37_projection_epsilon: float = 1e-12
    stage37_bc_weight: float = 0.1
    stage37_kl_weight: float = 0.1
    stage37_safety_kl_weight: float = 0.5
    stage37_optimizer_steps_per_epoch: int = 48
    stage37_gradient_accumulation: int = 8
    stage37_global_bucket_composition: tuple = (2, 30, 8, 24)
    stage37_diagnostic_only: bool = False
    stage37_jfi_num_members: int = 8
    stage37_jfi_checkpoint_path: str = ""
    stage37_jfi_calibration_path: str = ""
    stage37_jfi_train_manifest_path: str = ""
    stage37_jfi_candidate_bank_paths: Tuple[str, ...] = ()
    stage37_jfi_focal_gamma: float = 2.0
    stage37_jfi_positive_weight: float = 1.0
    stage37_jfi_joint_threshold: float = -1.0
    stage37_jfi_q10_floor: float = -1.0
    stage37_jfi_max_candidates: int = 4
    stage37_jfi_collect_calibration: bool = False
    # Stage38 elite-set counterfactual repair GRPO (frozen 2026-07-28).
    stage38_bucket_manifest_path: str = ""
    stage38_plan_sha256: str = (
        "8cd826b6ab8a4376c6e4a4fdffc97e96fe00fba00a25a1029d5fa5bcf4a8b027"
    )
    stage38_objective_revision: str = "elite_set_counterfactual_repair_v1"
    stage38_selector_micro_batch_size: int = 16
    stage38_elite_width: int = 5
    stage38_positive_challenger_width: int = 2
    stage38_active_pool_width: int = 8
    stage38_positive_margin: float = 0.001
    stage38_negative_tolerance: float = 0.0001
    stage38_advantage_scale: float = 0.002
    stage38_advantage_clip: float = 2.0
    stage38_mature_positive_multiplier: float = 0.25
    stage38_bc_weight: float = 0.1
    stage38_kl_weight: float = 0.1
    stage38_safety_kl_weight: float = 0.5
    stage38_step_discount: float = 0.6
    stage38_optimizer_steps_per_epoch: int = 48
    stage38_gradient_accumulation: int = 8
    stage38_global_bucket_composition: tuple = (2, 30, 8, 24)
    stage38_diagnostic_only: bool = False
    # Stage39 non-destructive challenger GRPO (frozen 2026-07-29).
    stage39_bucket_manifest_path: str = ""
    stage39_plan_sha256: str = (
        "7e1e6142412778524714a0b919f971edf1d2bf14138b0efcf21fc9dd7c4d6f1b"
    )
    stage39_objective_revision: str = (
        "non_destructive_challenger_set_grpo_v1"
    )
    stage39_public_candidate_count: int = 20
    stage39_challenger_candidate_count: int = 20
    stage39_group_size: int = 8
    stage39_elite_width: int = 5
    stage39_positive_margin: float = 0.001
    stage39_advantage_scale: float = 0.02
    stage39_advantage_clip: float = 2.0
    stage39_std_floor: float = 0.0001
    stage39_kl_weight: float = 0.1
    stage39_step_discount: float = 0.6
    stage39_optimizer_steps_per_epoch: int = 48
    stage39_gradient_accumulation: int = 8
    stage39_global_bucket_composition: tuple = (2, 30, 8, 24)
    stage39_candidate_source: str = "public20"
    stage39_challenger_noise_offset: int = 390001
    stage39_diagnostic_only: bool = False
    diffgrpo_lora_rank: int = 8
    diffgrpo_lora_alpha: float = 8.0
    generation_adaptive_kl_enabled: bool = False
    generation_kl_initial_coefficient: float = 0.1
    generation_kl_min_coefficient: float = 0.1
    generation_kl_max_coefficient: float = 100.0
    generation_kl_target: float = 1e-4
    generation_kl_hard_limit: float = 2.5e-4
    generation_kl_window: int = 32
    generation_kl_update_interval: int = 8
    generation_kl_adaptation_factor: float = 2.0
    generation_kl_lower_ratio: float = 2.0 / 3.0
    generation_kl_upper_ratio: float = 1.5
    generation_kl_hard_limit_patience: int = 2
    grpo_decoder_layer0_lr_mult: float = 1.0
    # group_zscore/reference_centered use K=1; within_anchor/hierarchical use
    # K=2; collision_truncated_intra_anchor also requires K=2;
    # anchor_hierarchical supports K=2 or K=4; anchor_rloo requires K=2.
    generation_advantage_mode: str = "group_zscore"
    reference_advantage_margin: float = 0.01
    reference_advantage_scale: float = 0.10
    reference_advantage_clip: float = 2.0
    rloo_advantage_margin: float = 0.01
    rloo_advantage_scale: float = 0.20
    rloo_advantage_clip: float = 1.0
    grpo_rollouts_per_mode: int = 1
    grpo_priority_manifest_path: str = ""
    grpo_priority_sample_fraction: float = 0.0
    generation_mode_weighting: str = "uniform"
    generation_mode_temperature: float = 1.0
    generation_ddim_eta: float = 1.0
    generation_final_std: float = 0.05
    generation_sigma_min: float = 1e-4
    generation_trust_projection_mode: str = "none"
    generation_trust_calibration_path: str = ""
    generation_trust_collect_calibration: bool = False
    generation_trust_calibration_seed: int = 20260719
    evaluation_noise_namespace: int = -1
    # Stage-0 audited schedule: the noise timestep matches the first DDIM step.
    diffusion_truncation_timestep: int = 8
    diffusion_roll_timesteps: Tuple[int, ...] = (8, 0)
    diffusion_scheduler_num_inference_steps: int = 125
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False
    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2


    # optmizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    # optimizer=dict(
    #     type="AdamW",
    #     lr=1e-4,
    #     weight_decay=1e-6,
    # )
    # scheduler=dict(
    #     type="MultiStepLR",
    #     milestones=[90],
    #     gamma=0.1,
    # )

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
