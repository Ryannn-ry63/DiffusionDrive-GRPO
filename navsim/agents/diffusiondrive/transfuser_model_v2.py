from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import copy
import hashlib
import json
import lzma
import math
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D, SinusoidalPosEmb
from navsim.agents.diffusiondrive.modules.blocks import (
    linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention,
)
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from navsim.agents.diffusiondrive.diffusion_grpo import (
    diagonal_gaussian_kl_same_std,
    _decode_policy_step,
    collect_full_chain_diffgrpo_trace,
    collect_paired_residual_diffgrpo_trace,
    collect_selected_anchor_diffgrpo_trace,
    collect_selected_set_diffgrpo_trace,
    collect_deployed_selected_set_diffgrpo_trace,
    collect_selector_aware_frontier_diffgrpo_trace,
    collect_mode_coverage_diffgrpo_trace,
    collect_generation_trace,
    compute_pdm_dense_rewards,
    compute_pdm_tiebreak_rewards,
    ddim_transition_with_log_prob,
    flatten_generation_rollouts,
    load_generation_trust_calibration,
    normalized_rms_displacement,
    project_reference_mean_ball,
    select_inference_mode,
    summarize_trust_projection,
    validate_inference_selector_source,
    validate_generation_trust_provenance,
)
from navsim.agents.diffusiondrive.trajectory_value_selector import (
    TrajectoryValueSelector,
    deterministic_bootstrap_mask,
    select_conservative_top2,
)
from navsim.agents.diffusiondrive.stage23_trajectory_selector import (
    Stage23TrajectorySelector,
    load_stage23_candidate_banks,
    load_stage23_token_log_map,
    member_training_mask,
    select_stage23_trajectory,
)
from navsim.agents.diffusiondrive.stage24_safety_value_selector import (
    Stage24SafetyValueSelector,
    compute_stage24_positive_weights,
    load_stage24_candidate_banks,
    load_stage24_token_log_map,
    select_stage24_trajectory,
    stage24_member_training_mask,
)
from navsim.agents.diffusiondrive.stage25_relative_harm_selector import (
    Stage25RelativeHarmSelector,
    compute_stage25_positive_weights,
    select_stage25_trajectory,
)
from navsim.agents.diffusiondrive.paired_advantage_risk import (
    PairedAdvantageRiskHead,
    select_same_mode_with_base_fallback,
)
from navsim.agents.diffusiondrive.stage30_mode_coverage import (
    load_stage30_bucket_manifest,
)
from navsim.agents.diffusiondrive.stage34_trace import (
    build_stage34_trace_diagnostics,
)
from navsim.agents.diffusiondrive.stage35_nested_trace import (
    collect_nested_counterfactual_deployment_trace,
)
from navsim.agents.diffusiondrive.stage36_trace import (
    collect_stage36_sampling_trace,
    finalize_stage36_reward_dependent_replay,
)
from navsim.agents.diffusiondrive.stage37_trace import (
    collect_stage37_sampling_trace,
    finalize_stage37_reward_dependent_replay,
)
from navsim.agents.diffusiondrive.stage38_trace import (
    collect_stage38_sampling_trace,
    finalize_stage38_reward_dependent_replay,
)
from navsim.agents.diffusiondrive.stage39_contract import (
    STAGE39_TRAINING_MODES,
)
from navsim.agents.diffusiondrive.stage39_trace import (
    collect_stage39_sampling_trace,
    finalize_stage39_replay,
)
from navsim.agents.diffusiondrive.stage37_joint_feasible_selector import (
    Stage37JointFeasibleSelector,
    select_stage37_jfi_trajectory,
)

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


STAGE19_BASE_SHA256 = (
    "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
)
STAGE19_FULL_SCHEDULE = {
    "truncation_timestep": 32,
    "roll_timesteps": [32, 24, 16, 8, 0],
    "scheduler_num_inference_steps": 125,
    "scheduler_step_stride": 8,
    "transitions": [[32, 24], [24, 16], [16, 8], [8, 0], [0, -8]],
}
SELECTED_SET_TRAINING_MODES = {
    "diffgrpo_selected_set",
    "stage27_public_diffgrpo_selected_set",
    "stage28_public_paired_uplift_multi",
    "stage28_public_paired_uplift_explore",
    "stage29_public_headroom_hybrid",
    "stage29_public_headroom_conditional",
}
MODE_COVERAGE_TRAINING_MODES = {
    "stage30_public_mode_coverage",
    "stage30_public_mode_coverage_constrained",
}
DEPLOYED_SET_TRAINING_MODES = {
    "stage31_public_deployed_pair",
    "stage31_public_deployed_frontier",
    "stage32_public_deployed_extended",
    # Stage33 CDC pilot reuses the selected-set common-noise trace until the
    # candidate-level collector is enabled by its manifest.
    "stage33_cdc_grpo",
}
STAGE32_FRONTIER_TRAINING_MODES = {
    "stage32_selector_aware_frontier",
}
STAGE34_MODE_ALIGNED_TRAINING_MODES = {
    "stage34_mode_aligned_frontier_grpo",
}
STAGE35_NCD_TRAINING_MODES = {
    "stage35_nested_counterfactual_deployment_grpo",
}
STAGE36_RGT_NCD_TRAINING_MODES = {
    "stage36_reference_gated_tail_ncd_grpo",
}
STAGE37_BPD_TRAINING_MODES = {
    "stage37_bistate_projected_deployment_grpo",
}
STAGE38_ESCR_TRAINING_MODES = {
    "stage38_elite_set_counterfactual_repair_grpo",
}
STAGE39_CHALLENGER_TRAINING_MODES = set(STAGE39_TRAINING_MODES)

def resolve_mode_coverage_bucket_manifest(config, training_mode: str):
    """Resolve the frozen bucket manifest owned by the active training stage."""
    if training_mode in STAGE39_CHALLENGER_TRAINING_MODES:
        stage = "Stage39"
        path = str(getattr(config, "stage39_bucket_manifest_path", ""))
    elif training_mode in STAGE38_ESCR_TRAINING_MODES:
        stage = "Stage38"
        path = str(getattr(config, "stage38_bucket_manifest_path", ""))
    elif training_mode in STAGE37_BPD_TRAINING_MODES:
        stage = "Stage37"
        path = str(getattr(config, "stage37_bucket_manifest_path", ""))
    elif training_mode in STAGE36_RGT_NCD_TRAINING_MODES:
        stage = "Stage36"
        path = str(getattr(config, "stage36_bucket_manifest_path", ""))
    elif training_mode in STAGE35_NCD_TRAINING_MODES:
        stage = "Stage35"
        path = str(getattr(config, "stage35_bucket_manifest_path", ""))
    elif training_mode in STAGE34_MODE_ALIGNED_TRAINING_MODES:
        stage = "Stage34"
        path = str(getattr(config, "stage34_bucket_manifest_path", ""))
    else:
        stage = "Stage30"
        path = str(getattr(config, "stage30_bucket_manifest_path", ""))
    if not path:
        raise ValueError(f"{stage} requires its frozen bucket manifest")
    return stage, path




def load_selected_anchor_modes(path: str, expected_group_size: int) -> Dict[str, int]:
    """Load a fail-closed Stage19 token-to-base-mode manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage19 selected-mode manifest missing: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if summary.get("base_checkpoint_sha256") != STAGE19_BASE_SHA256:
        raise RuntimeError("Stage19 manifest base checkpoint SHA mismatch")
    if summary.get("schedule") != STAGE19_FULL_SCHEDULE:
        raise RuntimeError("Stage19 manifest schedule mismatch")
    if summary.get("group_definition") != "same_token_same_selected_anchor":
        raise RuntimeError("Stage19 manifest group definition mismatch")
    if int(summary.get("group_size", -1)) != int(expected_group_size):
        raise RuntimeError("Stage19 manifest group size mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Stage19 manifest has no records")
    result = {}
    for record in records:
        token = str(record["token"])
        mode = int(record["selected_mode"])
        if token in result or mode < 0 or mode >= 20:
            raise RuntimeError(f"invalid Stage19 selected-mode record: {token}")
        result[token] = mode
    return result


def load_base_preserve_manifest(path: str, expected_group_size: int) -> Dict[str, Dict[str, Any]]:
    """Load the fail-closed Stage21 base-relative training metadata."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage21 base-preserve manifest missing: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    expected = {
        "base_checkpoint_sha256": STAGE19_BASE_SHA256,
        "schedule": STAGE19_FULL_SCHEDULE,
        "group_definition": "same_token_same_selected_anchor",
        "group_size": int(expected_group_size),
        "stage21_objective": "base_anchored_log_robust_v1",
        "base_noise_namespaces": [-1, 20260728],
    }
    for key, wanted in expected.items():
        if summary.get(key) != wanted:
            raise RuntimeError(
                f"Stage21 manifest {key} mismatch: {summary.get(key)!r} != {wanted!r}"
            )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Stage21 manifest has no records")
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        token = str(record["token"])
        mode = int(record["selected_mode"])
        base_reward = float(record["base_reward_mean"])
        base_safety = tuple(
            float(record[name])
            for name in (
                "base_collision_mean",
                "base_drivable_mean",
                "base_ttc_mean",
            )
        )
        scene_weight = float(record["scene_weight"])
        bc_weight = float(record["bc_weight"])
        finite = all(
            math.isfinite(value)
            for value in (base_reward, *base_safety, scene_weight, bc_weight)
        )
        if (
            token in result
            or mode < 0
            or mode >= 20
            or not finite
            or not 0.0 <= base_reward <= 1.0
            or any(not 0.0 <= value <= 1.0 for value in base_safety)
            or not 0.25 <= scene_weight <= 4.0
            or not 0.1 <= bc_weight <= 0.3
        ):
            raise RuntimeError(f"invalid Stage21 base-preserve record: {token}")
        result[token] = {
            "selected_mode": mode,
            "base_reward": base_reward,
            "base_safety": base_safety,
            "scene_weight": scene_weight,
            "bc_weight": bc_weight,
        }
    return result


def load_paired_residual_manifest(
    path: str, expected_group_size: int
) -> Dict[str, Dict[str, Any]]:
    """Load fail-closed Stage22 selected anchors and inverse-log weights."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage22 paired-residual manifest missing: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    expected = {
        "base_checkpoint_sha256": STAGE19_BASE_SHA256,
        "schedule": STAGE19_FULL_SCHEDULE,
        "group_definition": "same_token_same_selected_anchor_common_random",
        "group_size": int(expected_group_size),
        "stage22_objective": "paired_delta_bootstrap_exact_kl_lora_v2",
        "scene_weight_definition": "inverse_log_frequency_clip_0.5_2_unit_mean",
    }
    for key, wanted in expected.items():
        if summary.get(key) != wanted:
            raise RuntimeError(
                f"Stage22 manifest {key} mismatch: {summary.get(key)!r} != {wanted!r}"
            )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Stage22 manifest has no records")
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        token = str(record["token"])
        mode = int(record["selected_mode"])
        scene_weight = float(record["scene_weight"])
        if (
            token in result
            or mode < 0
            or mode >= 20
            or not math.isfinite(scene_weight)
            or not 0.5 <= scene_weight <= 2.0
        ):
            raise RuntimeError(f"invalid Stage22 paired-residual record: {token}")
        result[token] = {
            "selected_mode": mode,
            "scene_weight": scene_weight,
        }
    mean_weight = sum(item["scene_weight"] for item in result.values()) / len(result)
    if not math.isclose(mean_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Stage22 scene weights are not unit mean: {mean_weight}")
    return result


class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
        proposal_sampling = TrajectorySampling(time_horizon=4.0,interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        scorer_config = PDMScorerConfig(
                    progress_weight=10.0,
                    ttc_weight=5.0,
                    comfortable_weight=2.0
                    )
        self.scorer = PDMScorer(proposal_sampling, scorer_config)
        

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=None,tokens_list=tokens_list)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        #ego_fut_mode=64,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
            #ego_fut_mode=64
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
        gradient_scope="last_layer",
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        if gradient_scope not in {"last_layer", "all_layers"}:
            raise ValueError(
                "grpo_decoder_gradient_scope must be last_layer or all_layers"
            )
        self.gradient_scope = gradient_scope

    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for layer_index, mod in enumerate(self.layers):
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            next_points = poses_reg[..., :2].clone()
            if self.gradient_scope == "last_layer" and layer_index < self.num_layers - 1:
                next_points = next_points.detach()
            traj_points = next_points
        return poses_reg_list, poses_cls_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._config = config
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20
        #self.ego_fut_mode = 64

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )


        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(
            diff_decoder_layer,
            2,
            gradient_scope=str(
                getattr(config, "grpo_decoder_gradient_scope", "last_layer")
            ),
        )
        stage39_active = (
            str(getattr(config, "grpo_training_mode", "")) in STAGE39_TRAINING_MODES
            or str(getattr(config, "generation_policy_algorithm", ""))
            == "diffgrpo_non_destructive_challenger"
        )
        self.stage39_challenger_decoder = (
            copy.deepcopy(self.diff_decoder) if stage39_active else None
        )
        self.value_selector = TrajectoryValueSelector(
            config, num_heads=int(getattr(config, "value_selector_num_heads", 3))
        )
        self.register_buffer(
            "_value_selector_training_updates", torch.tensor(0, dtype=torch.long)
        )
        self.stage23_selector = Stage23TrajectorySelector(config)
        self.register_buffer(
            "_stage23_selector_training_updates", torch.tensor(0, dtype=torch.long)
        )
        self.stage24_selector = Stage24SafetyValueSelector(config)
        self.register_buffer(
            "_stage24_selector_training_updates", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "_stage24_ood_mean", torch.zeros(self.stage24_selector.selector_dim)
        )
        self.register_buffer(
            "_stage24_ood_variance", torch.ones(self.stage24_selector.selector_dim)
        )
        self.register_buffer(
            "_stage24_calibration_loaded", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer(
            "_stage24_embedding_count", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer(
            "_stage24_embedding_sum", torch.zeros(self.stage24_selector.selector_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "_stage24_embedding_sum_sq", torch.zeros(self.stage24_selector.selector_dim, dtype=torch.float64)
        )
        self.stage25_selector = Stage25RelativeHarmSelector(config)
        self.register_buffer(
            "_stage25_selector_training_updates", torch.tensor(0, dtype=torch.long)
        )
        self.stage37_jfi_selector = Stage37JointFeasibleSelector(config)
        self.register_buffer(
            "_stage37_jfi_training_updates", torch.tensor(0, dtype=torch.long)
        )
        # Optional adapters must not perturb legacy decoder/training RNG streams.
        with torch.random.fork_rng(devices=[]):
            self.paired_risk_head = PairedAdvantageRiskHead(config)
        self.register_buffer(
            "_paired_risk_training_updates", torch.tensor(0, dtype=torch.long)
        )
        if int(getattr(config, "value_selector_top_k", 2)) != 2:
            raise ValueError("Stage-15 value selector requires top_k=2")
        self.ref_policy = None
        self.old_policy = None
        self._old_policy_sync_steps = int(getattr(config, "grpo_old_policy_sync_steps", 32))
        self.register_buffer(
            "_old_policy_last_sync_step", torch.tensor(-1, dtype=torch.long)
        )
        self._truncation_timestep = int(
            getattr(config, "diffusion_truncation_timestep", 8)
        )
        self._roll_timesteps = tuple(
            int(t) for t in getattr(config, "diffusion_roll_timesteps", (8, 0))
        )
        self._scheduler_num_inference_steps = int(
            getattr(config, "diffusion_scheduler_num_inference_steps", 125)
        )
        self._grpo_training_mode = getattr(config, "grpo_training_mode", "classification_shared")
        self._inference_selector_source = validate_inference_selector_source(
            getattr(config, "inference_selector_source", "current")
        )
        self._grpo_scene_weight_mode = str(
            getattr(config, "grpo_scene_weight_mode", "uniform")
        )
        if self._grpo_scene_weight_mode not in {"uniform", "reference_headroom"}:
            raise ValueError(
                "grpo_scene_weight_mode must be one of "
                "{'uniform', 'reference_headroom'}; "
                f"got {self._grpo_scene_weight_mode!r}"
            )
        self._generation_ddim_eta = float(getattr(config, "generation_ddim_eta", 1.0))
        self._generation_final_std = float(getattr(config, "generation_final_std", 0.05))
        self._generation_sigma_min = float(getattr(config, "generation_sigma_min", 1e-4))
        self._generation_policy_algorithm = str(
            getattr(config, "generation_policy_algorithm", "legacy_ppo")
        )
        if self._generation_policy_algorithm not in {
            "legacy_ppo", "diffgrpo_full_chain", "diffgrpo_selected_anchor",
            "diffgrpo_selected_anchor_base_preserve", "diffgrpo_paired_residual",
            "diffgrpo_selected_set", "diffgrpo_mode_coverage",
            "diffgrpo_deployed_selected_set", "diffgrpo_mode_aligned_frontier",
            "diffgrpo_nested_counterfactual_deployment",
            "diffgrpo_reference_gated_tail_ncd",
            "diffgrpo_bistate_projected_deployment",
            "diffgrpo_elite_set_counterfactual_repair",
            "diffgrpo_non_destructive_challenger",
        }:
            raise ValueError(
                "generation_policy_algorithm must be legacy_ppo, "
                "diffgrpo_full_chain, diffgrpo_selected_anchor, or "
                "diffgrpo_selected_anchor_base_preserve/diffgrpo_paired_residual, "
                "or diffgrpo_selected_set/diffgrpo_mode_coverage/"
                "diffgrpo_deployed_selected_set/diffgrpo_mode_aligned_frontier"
                "/diffgrpo_nested_counterfactual_deployment"
                "/diffgrpo_reference_gated_tail_ncd"
                "/diffgrpo_bistate_projected_deployment"
                "/diffgrpo_elite_set_counterfactual_repair"
            )
        self._diffgrpo_bc_weight = float(getattr(config, "diffgrpo_bc_weight", 0.1))
        self._diffgrpo_step_discount = float(
            getattr(config, "diffgrpo_step_discount", 0.6)
        )
        self._diffgrpo_logprob_reduction = str(
            getattr(config, "diffgrpo_logprob_reduction", "mean")
        )
        self._generation_trust_projection_mode = str(
            getattr(config, "generation_trust_projection_mode", "none")
        )
        if self._generation_trust_projection_mode not in {
            "none",
            "reference_mean_ball",
        }:
            raise ValueError(
                "generation_trust_projection_mode must be none or "
                "reference_mean_ball"
            )
        self._generation_trust_collect_calibration = bool(
            getattr(config, "generation_trust_collect_calibration", False)
        )
        self._generation_trust_calibration_seed = int(
            getattr(config, "generation_trust_calibration_seed", 20260719)
        )
        self._evaluation_noise_namespace = int(
            getattr(config, "evaluation_noise_namespace", -1)
        )
        if self._evaluation_noise_namespace < -1:
            raise ValueError("evaluation_noise_namespace must be -1 or non-negative")
        if (
            self._generation_trust_collect_calibration
            and self._generation_trust_projection_mode != "none"
        ):
            raise ValueError(
                "calibration collection requires generation trust mode=none"
            )
        self._generation_trust_calibration = None
        self._generation_trust_radii = {}
        calibration_path = str(
            getattr(config, "generation_trust_calibration_path", "")
        )
        if self._generation_trust_projection_mode == "reference_mean_ball":
            if not calibration_path:
                raise ValueError(
                    "reference_mean_ball requires generation_trust_calibration_path"
                )
            self._generation_trust_calibration = (
                load_generation_trust_calibration(calibration_path)
            )
            self._generation_trust_radii = {
                step: float(self._generation_trust_calibration["steps"][step]["radius"])
                for step in ("transition", "final")
            }
        self._generation_advantage_mode = str(
            getattr(config, "generation_advantage_mode", "group_zscore")
        )
        if self._generation_advantage_mode not in {
            "group_zscore",
            "reference_centered",
            "within_anchor",
            "hierarchical",
            "anchor_hierarchical",
            "anchor_rloo",
            "collision_truncated_intra_anchor",
            "escr_set_marginal",
        }:
            raise ValueError(
                "generation_advantage_mode must be one of "
                "{'group_zscore', 'reference_centered', "
                "'within_anchor', 'hierarchical', 'anchor_hierarchical', "
                "'anchor_rloo', 'collision_truncated_intra_anchor', "
                "'escr_set_marginal'}; "
                f"got {self._generation_advantage_mode!r}"
            )
        self._grpo_rollouts_per_mode = int(
            getattr(config, "grpo_rollouts_per_mode", 1)
        )
        if self._grpo_rollouts_per_mode not in {1, 2, 4}:
            raise ValueError(
                "grpo_rollouts_per_mode must be 1, 2, or 4"
            )
        legacy_hierarchical_mode = self._generation_advantage_mode in {
            "within_anchor",
            "hierarchical",
            "collision_truncated_intra_anchor",
        }
        if legacy_hierarchical_mode and self._grpo_rollouts_per_mode != 2:
            raise ValueError(
                f"{self._generation_advantage_mode} requires "
                "grpo_rollouts_per_mode=2; "
                f"got {self._grpo_rollouts_per_mode}"
            )
        if (
            self._generation_advantage_mode == "anchor_hierarchical"
            and self._grpo_rollouts_per_mode not in {2, 4}
        ):
            raise ValueError(
                "anchor_hierarchical requires grpo_rollouts_per_mode=2 or 4"
            )
        if (
            self._generation_advantage_mode == "anchor_rloo"
            and self._grpo_rollouts_per_mode != 2
        ):
            raise ValueError(
                "anchor_rloo requires grpo_rollouts_per_mode=2"
            )
        if (
            self._generation_advantage_mode
            in {"group_zscore", "reference_centered"}
            and self._grpo_rollouts_per_mode != 1
        ):
            raise ValueError(
                "group_zscore/reference_centered require "
                "grpo_rollouts_per_mode=1"
            )
        self._diffgrpo_group_size = int(
            getattr(config, "diffgrpo_group_size", 8)
        )
        self._selected_anchor_modes = {}
        self._base_preserve_metadata = {}
        self._paired_residual_metadata = {}
        self._stage30_bucket_by_token = {}
        self._stage23_token_logs = {}
        self._stage23_candidate_bank = {}
        stage23_manifest = str(
            getattr(config, "stage23_selector_train_manifest_path", "")
        )
        if self._grpo_training_mode == "stage23_selector":
            if not stage23_manifest:
                raise ValueError("stage23_selector requires its train manifest")
            self._stage23_token_logs = load_stage23_token_log_map(stage23_manifest)
            bank_paths = tuple(getattr(config, "stage23_candidate_bank_paths", ()))
            self._stage23_candidate_bank = load_stage23_candidate_banks(
                bank_paths, STAGE19_BASE_SHA256
            )
            if set(self._stage23_candidate_bank) != set(self._stage23_token_logs):
                raise RuntimeError(
                    "Stage23 candidate-bank tokens differ from the fit manifest"
                )
        self._stage24_token_logs = {}
        self._stage24_candidate_bank = {}
        self._stage24_bank_combinations = ()
        self._stage24_positive_weights = ((1.0, 1.0, 1.0), 1.0)
        self._stage25_positive_weights = (1.0, 1.0, 1.0, 1.0)
        stage24_manifest = str(getattr(
            config,
            (
                "stage37_jfi_train_manifest_path"
                if self._grpo_training_mode == "stage37_jfi_selector"
                else "stage24_selector_train_manifest_path"
            ),
            "",
        ))
        if self._grpo_training_mode in {
            "stage24_selector", "stage25_relative_harm_selector",
            "stage37_jfi_selector",
        }:
            if not stage24_manifest:
                raise ValueError("Stage24/JFI selector requires its train manifest")
            self._stage24_token_logs = load_stage24_token_log_map(stage24_manifest)
            bank_paths = tuple(getattr(
                config,
                (
                    "stage37_jfi_candidate_bank_paths"
                    if self._grpo_training_mode == "stage37_jfi_selector"
                    else "stage24_candidate_bank_paths"
                ),
                (),
            ))
            (
                self._stage24_candidate_bank,
                self._stage24_bank_combinations,
            ) = load_stage24_candidate_banks(
                bank_paths, tuple(self._stage24_token_logs)
            )
            if set(self._stage24_candidate_bank) != set(self._stage24_token_logs):
                raise RuntimeError(
                    "Stage24 candidate-bank tokens differ from the train manifest"
                )
            self._stage24_positive_weights = compute_stage24_positive_weights(
                self._stage24_candidate_bank
            )
            self._stage25_positive_weights = compute_stage25_positive_weights(
                self._stage24_candidate_bank
            )
            if int(getattr(config, "stage24_selector_steps_per_epoch", 0)) <= 0:
                raise ValueError("Stage24 requires positive steps_per_epoch")
        selected_mode_manifest = str(
            getattr(config, "diffgrpo_selected_mode_manifest_path", "")
        )
        if self._grpo_training_mode == "diffgrpo_selected_anchor":
            self._selected_anchor_modes = load_selected_anchor_modes(
                selected_mode_manifest, self._diffgrpo_group_size
            )
        elif self._grpo_training_mode == "diffgrpo_selected_anchor_base_preserve":
            self._base_preserve_metadata = load_base_preserve_manifest(
                selected_mode_manifest, self._diffgrpo_group_size
            )
            self._selected_anchor_modes = {
                token: int(metadata["selected_mode"])
                for token, metadata in self._base_preserve_metadata.items()
            }
        elif self._grpo_training_mode == "diffgrpo_paired_residual":
            self._paired_residual_metadata = load_paired_residual_manifest(
                selected_mode_manifest, self._diffgrpo_group_size
            )
            self._selected_anchor_modes = {
                token: int(metadata["selected_mode"])
                for token, metadata in self._paired_residual_metadata.items()
            }
        if self._grpo_training_mode in (
            MODE_COVERAGE_TRAINING_MODES | STAGE34_MODE_ALIGNED_TRAINING_MODES
            | STAGE35_NCD_TRAINING_MODES | STAGE36_RGT_NCD_TRAINING_MODES
            | STAGE37_BPD_TRAINING_MODES | STAGE38_ESCR_TRAINING_MODES
            | STAGE39_CHALLENGER_TRAINING_MODES
        ):
            bucket_stage, bucket_manifest = resolve_mode_coverage_bucket_manifest(
                config, self._grpo_training_mode
            )
            self._stage30_bucket_by_token = load_stage30_bucket_manifest(
                bucket_manifest, require_full=True
            )
            self._mode_coverage_bucket_stage = bucket_stage
        self._grpo_reward_mode = str(getattr(config, "grpo_reward_mode", "pdms"))
        if self._grpo_reward_mode not in {"pdms", "pdm_tiebreak", "pdm_dense"}:
            raise ValueError(
                "grpo_reward_mode must be one of "
                "{'pdms', 'pdm_tiebreak', 'pdm_dense'}; "
                f"got {self._grpo_reward_mode!r}"
            )
        self._pdm_tiebreak_max_epsilon = float(
            getattr(config, "pdm_tiebreak_max_epsilon", 1e-3)
        )
        if self._pdm_tiebreak_max_epsilon <= 0:
            raise ValueError("pdm_tiebreak_max_epsilon must be positive")
        self._pdm_dense_weight = float(
            getattr(config, "pdm_dense_weight", 0.1)
        )
        if self._pdm_dense_weight < 0:
            raise ValueError("pdm_dense_weight must be non-negative")
        self._validate_roll_schedule()
        if self._grpo_training_mode == "diffgrpo_full_chain":
            if self._generation_policy_algorithm != "diffgrpo_full_chain":
                raise ValueError(
                    "diffgrpo_full_chain mode requires matching generation algorithm"
                )
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError(
                    "Stage-16 full-chain schedule must be [32,24,16,8,0]"
                )
            if self._truncation_timestep != 32:
                raise ValueError("Stage-16 truncation timestep must be 32")
            if self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage-16 scheduler inference steps must be 125")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage-16 requires group_zscore advantages")
            if self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage-16 requires one rollout per anchor mode")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage-16 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage-16 requires mean log-probability reduction")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage-16 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage-16 step discount must be 0.6")
        if self._grpo_training_mode == "diffgrpo_selected_anchor":
            if self._generation_policy_algorithm != "diffgrpo_selected_anchor":
                raise ValueError(
                    "selected-anchor mode requires matching generation algorithm"
                )
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage19 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32:
                raise ValueError("Stage19 truncation timestep must be 32")
            if self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage19 scheduler inference steps must be 125")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage19 requires group_zscore advantages")
            if self._diffgrpo_group_size != 8:
                raise ValueError("Stage19 requires selected-anchor group size 8")
            if self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage19 does not use legacy per-mode rollouts")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage19 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage19 requires mean log-probability reduction")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage19 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage19 step discount must be 0.6")
        if self._grpo_training_mode == "diffgrpo_selected_anchor_base_preserve":
            if self._generation_policy_algorithm != "diffgrpo_selected_anchor_base_preserve":
                raise ValueError("Stage21 mode requires its matching generation algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage21 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage21 requires truncation=32 and scheduler steps=125")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage21 requires group_zscore evidence")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage21 requires G=8 and one legacy rollout per mode")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage21 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage21 requires mean log-probability reduction")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage21 minimum BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage21 step discount must be 0.6")
        if self._grpo_training_mode == "diffgrpo_paired_residual":
            if self._generation_policy_algorithm != "diffgrpo_paired_residual":
                raise ValueError("Stage22 mode requires its matching algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage22 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage22 requires truncation=32 and scheduler steps=125")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage22 requires paired-delta group z-score")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage22 requires G=8")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage22 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage22 requires mean log-probability reduction")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage22 step discount must be 0.6")
        if self._grpo_training_mode in SELECTED_SET_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_selected_set":
                raise ValueError("Stage23 mode requires diffgrpo_selected_set")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage23 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage23 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage23 requires G=8 and legacy rollout multiplier 1")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage23 requires within-set group z-score")
            if self._inference_selector_source not in {
                "trajectory_oof", "trajectory_relative_harm_v3"
            }:
                raise ValueError(
                    "selected-set GRPO requires a frozen trajectory selector"
                )
            if self._inference_selector_source == "trajectory_oof":
                if float(getattr(
                    config, "stage23_selector_residual_margin", -1.0
                )) < 0:
                    raise ValueError(
                        "Stage23 generator requires calibrated selector margin"
                    )
            elif any(float(getattr(config, name, -1.0)) < 0 for name in (
                "stage24_selector_residual_margin",
                "stage24_selector_ood_threshold",
                "stage25_selector_risk_threshold",
            )):
                raise ValueError(
                    "Stage25 generator requires calibrated value, OOD, and risk "
                    "thresholds"
                )
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage23 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage23 requires mean log probabilities")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage23 BC weight must be 0.1")
        if self._grpo_training_mode in DEPLOYED_SET_TRAINING_MODES:
            if (
                self._generation_policy_algorithm
                != "diffgrpo_deployed_selected_set"
            ):
                raise ValueError(
                    "Stage31 requires diffgrpo_deployed_selected_set"
                )
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage31 schedule must be [32,24,16,8,0]")
            if (
                self._truncation_timestep != 32
                or self._scheduler_num_inference_steps != 125
            ):
                raise ValueError(
                    "Stage31 requires truncation=32 and scheduler steps=125"
                )
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage31 requires G=8")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage31 requires group_zscore configuration")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage31 requires the frozen S-multi selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage31 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage31 requires mean log probabilities")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage31 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage31 step discount must be 0.6")
        if self._grpo_training_mode in STAGE32_FRONTIER_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_deployed_selected_set":
                raise ValueError("Stage32 requires diffgrpo_deployed_selected_set")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage32 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage32 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage32 requires G=8")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage32 requires group_zscore configuration")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage32 requires the frozen S-multi selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage32 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage32 requires mean log probabilities")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage32 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage32 step discount must be 0.6")
            if int(getattr(config, "stage32_frontier_pool_size", -1)) != 4:
                raise ValueError("Stage32 requires frontier pool size=4")
        if self._grpo_training_mode in STAGE34_MODE_ALIGNED_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_mode_aligned_frontier":
                raise ValueError("Stage34 requires diffgrpo_mode_aligned_frontier")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage34 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage34 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 20:
                raise ValueError("Stage34 requires one complete 20-mode group")
            if self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage34 requires one rollout per anchor")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage34 requires the locked GRPO route")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage34 requires the frozen S-multi selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage34 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage34 requires mean log probabilities")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage34 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage34 step discount must be 0.6")
        if self._grpo_training_mode in STAGE35_NCD_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_nested_counterfactual_deployment":
                raise ValueError("Stage35 requires its nested counterfactual algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage35 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage35 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage35 requires G=8")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage35 requires the locked DPEL outer route")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage35 requires the frozen S-multi selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage35 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage35 requires mean log probabilities")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage35 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage35 step discount must be 0.6")
            if int(getattr(config, "stage35_active_pool_width", -1)) != 4:
                raise ValueError("Stage35 active pool width must be four")
            if int(getattr(config, "stage35_selector_micro_batch_size", 0)) <= 0:
                raise ValueError("Stage35 selector micro-batch must be positive")
            if float(getattr(config, "stage35_counterfactual_weight", 0.0)) <= 0:
                raise ValueError("Stage35 counterfactual weight must be positive")
        if self._grpo_training_mode in STAGE36_RGT_NCD_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_reference_gated_tail_ncd":
                raise ValueError("Stage36 requires its reference-gated tail algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage36 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage36 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage36 requires G=8")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage36 requires the locked DPEL outer route")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage36 requires the frozen S-multi selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage36 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage36 requires mean log probabilities")
            if abs(self._diffgrpo_bc_weight - 0.1) > 1e-12:
                raise ValueError("Stage36 BC weight must be 0.1")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage36 step discount must be 0.6")
            if int(getattr(config, "stage36_active_pool_width", -1)) != 4:
                raise ValueError("Stage36 active pool width must be four")
            if int(getattr(config, "stage36_selector_micro_batch_size", 0)) <= 0:
                raise ValueError("Stage36 selector micro-batch must be positive")
            if int(getattr(config, "stage36_frontier_width", -1)) != 5:
                raise ValueError("Stage36 public frontier width must be five")
            if int(getattr(config, "stage36_tail_elite_count", -1)) != 2:
                raise ValueError("Stage36 same-anchor elite count must be two")
        if self._grpo_training_mode in STAGE37_BPD_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_bistate_projected_deployment":
                raise ValueError("Stage37 requires its bi-state projected algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage37 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage37 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage37 requires G=8")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage37 requires the locked nested-deployment route")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage37 generator training requires frozen Stage25")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage37 does not permit inference trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage37 requires mean log probabilities")
            if int(getattr(config, "stage37_active_pool_width", -1)) != 4:
                raise ValueError("Stage37 active pool width must be four")
            if int(getattr(config, "stage37_frontier_width", -1)) != 5:
                raise ValueError("Stage37 public frontier width must be five")
            if int(getattr(config, "stage37_tail_elite_count", -1)) != 2:
                raise ValueError("Stage37 same-anchor elite count must be two")
        if self._grpo_training_mode in STAGE39_CHALLENGER_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_non_destructive_challenger":
                raise ValueError("Stage39 requires the non-destructive challenger algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage39 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage39 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage39 requires G=8 and one rollout per mode")
            if self._generation_advantage_mode != "stage39_branch_objective":
                raise ValueError("Stage39 requires stage39_branch_objective advantages")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage39 training requires the frozen Stage25 selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage39 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage39 requires mean log probabilities")
            if int(getattr(self._config, "stage39_public_candidate_count", -1)) != 20:
                raise ValueError("Stage39 public candidate count must be 20")
            if int(getattr(self._config, "stage39_challenger_candidate_count", -1)) != 20:
                raise ValueError("Stage39 challenger candidate count must be 20")
            if abs(float(getattr(self._config, "stage39_positive_margin", -1.0)) - 0.001) > 1e-12:
                raise ValueError("Stage39 positive margin must be 0.001")
            if abs(float(getattr(self._config, "stage39_kl_weight", -1.0)) - 0.1) > 1e-12:
                raise ValueError("Stage39 KL weight must be 0.1")
        if self._grpo_training_mode in STAGE38_ESCR_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_elite_set_counterfactual_repair":
                raise ValueError("Stage38 requires the ESCR generation algorithm")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage38 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage38 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 8 or self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage38 requires G=8 and one rollout per mode")
            if self._generation_advantage_mode != "escr_set_marginal":
                raise ValueError("Stage38 requires escr_set_marginal advantages")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage38 requires the frozen Stage25 selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage38 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage38 requires mean log probabilities")
            if int(getattr(config, "stage38_elite_width", -1)) != 5:
                raise ValueError("Stage38 public elite width must be five")
            if int(getattr(config, "stage38_positive_challenger_width", -1)) != 2:
                raise ValueError("Stage38 positive challenger width must be two")
            if int(getattr(config, "stage38_active_pool_width", -1)) != 8:
                raise ValueError("Stage38 active pool width must be eight")
            if abs(float(getattr(config, "stage38_bc_weight", -1.0)) - 0.1) > 1e-12:
                raise ValueError("Stage38 BC weight must be 0.1")
            if abs(float(getattr(config, "stage38_kl_weight", -1.0)) - 0.1) > 1e-12:
                raise ValueError("Stage38 active KL weight must be 0.1")
            if abs(float(getattr(config, "stage38_safety_kl_weight", -1.0)) - 0.5) > 1e-12:
                raise ValueError("Stage38 safety KL weight must be 0.5")
            if abs(float(getattr(config, "stage38_step_discount", -1.0)) - 0.6) > 1e-12:
                raise ValueError("Stage38 diffusion step discount must be 0.6")
        if self._grpo_training_mode in MODE_COVERAGE_TRAINING_MODES:
            if self._generation_policy_algorithm != "diffgrpo_mode_coverage":
                raise ValueError("Stage30 requires diffgrpo_mode_coverage")
            if self._roll_timesteps != (32, 24, 16, 8, 0):
                raise ValueError("Stage30 schedule must be [32,24,16,8,0]")
            if self._truncation_timestep != 32 or self._scheduler_num_inference_steps != 125:
                raise ValueError("Stage30 requires truncation=32 and scheduler steps=125")
            if self._diffgrpo_group_size != 20:
                raise ValueError("Stage30 requires one complete 20-mode group")
            if self._grpo_rollouts_per_mode != 1:
                raise ValueError("Stage30 requires one rollout per anchor")
            if self._generation_advantage_mode != "group_zscore":
                raise ValueError("Stage30 requires group_zscore configuration")
            if self._inference_selector_source != "trajectory_relative_harm_v3":
                raise ValueError("Stage30 requires the frozen S-multi selector")
            if self._generation_trust_projection_mode != "none":
                raise ValueError("Stage30 does not permit trust projection")
            if self._diffgrpo_logprob_reduction != "mean":
                raise ValueError("Stage30 requires mean log probabilities")
            if abs(self._diffgrpo_step_discount - 0.6) > 1e-12:
                raise ValueError("Stage23 step discount must be 0.6")

        self.loss_computer = LossComputer(config)

        proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        scorer_config = PDMScorerConfig(
            progress_weight=10.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
        )
        self.scorer = PDMScorer(proposal_sampling, scorer_config)

        # Lazy-load metric caches on first use (no long startup). Optionally LRU-evict when metric_cache_lru_max > 0.
        self.metric_cache_loader: Optional[MetricCacheLoader] = None
        self._metric_cache_lru_max: int = int(getattr(config, "metric_cache_lru_max", 0))
        self._lazy_metric_cache: Optional[Union[Dict[str, Any], OrderedDict]] = None
        metric_cache_dir = getattr(config, "metric_cache_path", "")
        if metric_cache_dir and Path(metric_cache_dir).exists():
            self.metric_cache_loader = MetricCacheLoader(Path(metric_cache_dir))
            self._lazy_metric_cache = OrderedDict() if self._metric_cache_lru_max > 0 else {}
            n_paths = len(self.metric_cache_loader.metric_cache_paths)
            print(
                f"Metric cache: lazy load enabled ({n_paths} tokens on disk; "
                f"loads on demand during training, no startup preload)."
                + (f" LRU max entries: {self._metric_cache_lru_max}." if self._metric_cache_lru_max > 0 else "")
            )

        self._reward_compute_interval: int = int(getattr(config, "reward_compute_interval", 1))
        if self._reward_compute_interval != 1:
            raise ValueError(
                "reward_compute_interval must be 1 because rewards depend on current trajectories"
            )

    def _get_metric_cache_lazy(self, token: str) -> Any:
        """Return metric cache for token, loading from disk on first miss."""
        assert self._lazy_metric_cache is not None and self.metric_cache_loader is not None
        store = self._lazy_metric_cache
        if token in store:
            if isinstance(store, OrderedDict):
                store.move_to_end(token)
            return store[token]
        path = self.metric_cache_loader.metric_cache_paths.get(token)
        if path is None:
            return None
        try:
            with lzma.open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            return None
        store[token] = obj
        if isinstance(store, OrderedDict):
            store.move_to_end(token)
            if self._metric_cache_lru_max > 0:
                while len(store) > self._metric_cache_lru_max:
                    store.popitem(last=False)
        return obj

    def set_ref_policy(self, ref_policy):
        self.ref_policy = ref_policy
        print("TrajectoryHead: Reference policy set (frozen pretrained weights)")    

    def set_old_policy(self, old_policy):
        self.old_policy = old_policy
        print("TrajectoryHead: Old policy set (periodically synchronized snapshot)")

    def _validate_roll_schedule(self) -> None:
        """Fail early for invalid or ambiguous truncated-DDIM schedules."""
        num_train_timesteps = self.diffusion_scheduler.config.num_train_timesteps
        if not self._roll_timesteps:
            raise ValueError("diffusion_roll_timesteps must contain at least one timestep")
        if any(t < 0 or t >= num_train_timesteps for t in self._roll_timesteps):
            raise ValueError(
                "diffusion_roll_timesteps must be within the scheduler training range"
            )
        if any(a <= b for a, b in zip(self._roll_timesteps, self._roll_timesteps[1:])):
            raise ValueError("diffusion_roll_timesteps must be strictly decreasing")
        if self._truncation_timestep < 0:
            raise ValueError("diffusion_truncation_timestep must be non-negative")
        if self._scheduler_num_inference_steps <= 0:
            raise ValueError("diffusion_scheduler_num_inference_steps must be positive")

    def get_roll_schedule(self):
        """Return the explicit schedule used by training and evaluation."""
        step_stride = (
            self.diffusion_scheduler.config.num_train_timesteps
            // self._scheduler_num_inference_steps
        )
        return {
            "truncation_timestep": self._truncation_timestep,
            "roll_timesteps": self._roll_timesteps,
            "scheduler_num_inference_steps": self._scheduler_num_inference_steps,
            "scheduler_step_stride": step_stride,
            "transitions": tuple((t, t - step_stride) for t in self._roll_timesteps),
        }

    def get_generation_trust_sigmas(self) -> Dict[str, float]:
        """Return the exact normalized-space standard deviations used by trust balls."""
        if len(self._roll_timesteps) != 2:
            raise ValueError("generation trust projection requires exactly two steps")
        stride = (
            self.diffusion_scheduler.config.num_train_timesteps
            // self._scheduler_num_inference_steps
        )
        first_timestep = self._roll_timesteps[0]
        variance = self.diffusion_scheduler._get_variance(
            int(first_timestep), int(first_timestep) - stride
        )
        transition_sigma = max(
            self._generation_sigma_min,
            self._generation_ddim_eta * float(variance.sqrt()),
        )
        return {
            "transition": transition_sigma,
            "final": self._generation_final_std,
        }

    def validate_generation_trust_runtime(
        self, reference_checkpoint_path: str
    ) -> None:
        """Bind a loaded radius artifact to the actual frozen base checkpoint."""
        if self._generation_trust_projection_mode == "none":
            return
        if self.ref_policy is None or self._generation_trust_calibration is None:
            raise RuntimeError("Hard trust projection has no frozen base/calibration")
        sigmas = self.get_generation_trust_sigmas()
        validate_generation_trust_provenance(
            self._generation_trust_calibration,
            reference_checkpoint_path,
            self._roll_timesteps,
            self._scheduler_num_inference_steps,
            sigmas["transition"],
            sigmas["final"],
        )

    @torch.no_grad()
    def maybe_sync_old_policy(self, optimizer_step: int):
        """Synchronize PPO behavior policy on optimizer-step boundaries.

        The persistent last-sync step keeps old policy and cadence continuous
        across a Lightning checkpoint resume.
        """
        if self.old_policy is None:
            return
        optimizer_step = int(optimizer_step)
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        last_sync = int(self._old_policy_last_sync_step.item())
        if last_sync < 0 or optimizer_step - last_sync >= self._old_policy_sync_steps:
            self.old_policy.load_state_dict(self.diff_decoder.state_dict())
            self.old_policy.eval()
            self._old_policy_last_sync_step.fill_(optimizer_step)
    
    
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def _lookup_selected_anchor_modes(self, tokens_list, device):
        if tokens_list is None:
            raise RuntimeError("Stage19 training requires scene tokens")
        missing = [
            str(token)
            for token in tokens_list
            if str(token) not in self._selected_anchor_modes
        ]
        if missing:
            raise RuntimeError(
                f"Stage19 manifest has no selected mode for token {missing[0]}"
            )
        return torch.tensor(
            [self._selected_anchor_modes[str(token)] for token in tokens_list],
            dtype=torch.long,
            device=device,
        )

    def _lookup_base_preserve_metadata(self, tokens_list, device):
        if tokens_list is None:
            raise RuntimeError("Stage21 training requires scene tokens")
        missing = [
            str(token)
            for token in tokens_list
            if str(token) not in self._base_preserve_metadata
        ]
        if missing:
            raise RuntimeError(
                f"Stage21 manifest has no base metadata for token {missing[0]}"
            )
        records = [self._base_preserve_metadata[str(token)] for token in tokens_list]
        return {
            "base_rewards": torch.tensor(
                [record["base_reward"] for record in records],
                dtype=torch.float32,
                device=device,
            ),
            "base_safety": torch.tensor(
                [record["base_safety"] for record in records],
                dtype=torch.float32,
                device=device,
            ),
            "scene_weights": torch.tensor(
                [record["scene_weight"] for record in records],
                dtype=torch.float32,
                device=device,
            ),
            "bc_weights": torch.tensor(
                [record["bc_weight"] for record in records],
                dtype=torch.float32,
                device=device,
            ),
        }

    def _lookup_paired_residual_weights(self, tokens_list, device):
        if tokens_list is None:
            raise RuntimeError("Stage22 training requires scene tokens")
        missing = [
            str(token) for token in tokens_list
            if str(token) not in self._paired_residual_metadata
        ]
        if missing:
            raise RuntimeError(
                f"Stage22 manifest has no metadata for token {missing[0]}"
            )
        return torch.tensor(
            [
                self._paired_residual_metadata[str(token)]["scene_weight"]
                for token in tokens_list
            ],
            dtype=torch.float32,
            device=device,
        )

    def _lookup_stage30_buckets(self, tokens_list, device):
        if tokens_list is None:
            stage = getattr(self, "_mode_coverage_bucket_stage", "Stage30")
            raise RuntimeError(f"{stage} training requires scene tokens")
        missing = [
            str(token) for token in tokens_list
            if str(token) not in self._stage30_bucket_by_token
        ]
        if missing:
            stage = getattr(self, "_mode_coverage_bucket_stage", "Stage30")
            raise RuntimeError(
                f"{stage} bucket manifest has no token {missing[0]}"
            )
        return torch.tensor(
            [self._stage30_bucket_by_token[str(token)] for token in tokens_list],
            dtype=torch.long,
            device=device,
        )
    
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,tokens_list=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train_grpo(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,tokens_list)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img,tokens_list)


    def _run_policy_rollout(
        self, policy, initial_sample, ego_query, agents_query, bev_feature,
        bev_spatial_shape, status_encoding, global_img,
    ):
        """Run the same two-step truncated DDIM chain used at inference."""
        bs = initial_sample.shape[0]
        device = initial_sample.device
        self.diffusion_scheduler.set_timesteps(
            self._scheduler_num_inference_steps, device
        )
        roll_timesteps = torch.tensor(
            self._roll_timesteps, device=device, dtype=torch.long
        )
        sample = initial_sample

        for timestep in roll_timesteps:
            noisy_traj_points = self.denorm_odo(sample.clamp(min=-1, max=1))
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed.flatten(-2))
            traj_feature = traj_feature.view(bs, noisy_traj_points.shape[1], -1)
            time_embed = self.time_mlp(timestep.expand(bs)).view(bs, 1, -1)
            poses_reg_list, poses_cls_list = policy(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            sample = self.diffusion_scheduler.step(
                model_output=self.norm_odo(poses_reg[..., :2]),
                timestep=timestep,
                sample=sample,
            ).prev_sample

        return poses_reg, poses_cls

    def _run_policy_rollout_with_reference(
        self,
        initial_sample,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        apply_projection: bool,
    ):
        """Run current/base on identical states and optionally project each mean."""
        if self.ref_policy is None:
            raise RuntimeError("Trust rollout requires a frozen base policy")
        if len(self._roll_timesteps) != 2 or self._roll_timesteps[-1] != 0:
            raise ValueError("Trust rollout requires transition and final timesteps")
        self.diffusion_scheduler.set_timesteps(
            self._scheduler_num_inference_steps, initial_sample.device
        )
        sigmas = self.get_generation_trust_sigmas()
        sample = initial_sample
        diagnostics_by_step = []
        final_reg = final_cls = final_reference_cls = None

        for step_index, timestep in enumerate(self._roll_timesteps):
            current_reg, current_cls = _decode_policy_step(
                self, self.diff_decoder, sample, timestep, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            with torch.no_grad():
                reference_reg, reference_cls = _decode_policy_step(
                    self, self.ref_policy, sample.detach(), timestep,
                    ego_query.detach(), agents_query.detach(), bev_feature.detach(),
                    bev_spatial_shape, status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
            step_name = "transition" if step_index == 0 else "final"
            sigma = current_reg.new_tensor(sigmas[step_name]).detach()
            if step_name == "transition":
                current_mean = self.diffusion_scheduler.step(
                    model_output=self.norm_odo(current_reg[..., :2]),
                    timestep=timestep,
                    sample=sample,
                ).prev_sample
                with torch.no_grad():
                    reference_mean = self.diffusion_scheduler.step(
                        model_output=self.norm_odo(reference_reg[..., :2]),
                        timestep=timestep,
                        sample=sample.detach(),
                    ).prev_sample.detach()
            else:
                current_mean = self.norm_odo(current_reg)
                reference_mean = self.norm_odo(reference_reg).detach()

            if apply_projection:
                projected_mean, diagnostics = project_reference_mean_ball(
                    current_mean,
                    reference_mean,
                    sigma,
                    self._generation_trust_radii[step_name],
                )
            else:
                distance = normalized_rms_displacement(
                    current_mean, reference_mean, sigma
                ).detach()
                projected_mean = current_mean
                diagnostics = {
                    "pre_distance": distance,
                    "post_distance": distance,
                    "alpha": torch.ones_like(distance),
                    "projected": torch.zeros_like(distance, dtype=torch.bool),
                    "reference_coverage": torch.ones_like(distance, dtype=torch.bool),
                }
            diagnostics_by_step.append(diagnostics)
            if step_name == "transition":
                sample = projected_mean
            else:
                final_reg = (
                    self.denorm_odo(projected_mean)
                    if apply_projection
                    else current_reg
                )
                final_cls = current_cls
                final_reference_cls = reference_cls

        diagnostics = {
            f"generation_trust_{key}": torch.stack(
                [step[key] for step in diagnostics_by_step], dim=-1
            )
            for key in (
                "pre_distance",
                "post_distance",
                "alpha",
                "projected",
                "reference_coverage",
            )
        }
        return final_reg, final_cls, final_reference_cls, diagnostics

    def _compute_reference_selected_reward(
        self,
        final_poses_reg,
        final_ref_poses_cls,
        raw_rewards,
        reward_valid_mask,
        initial_sample,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        tokens_list,
    ):
        """Score the fixed-reference deployed choice without policy gradients."""
        if self._grpo_training_mode in {"generation", "joint"}:
            with torch.no_grad():
                reference_reg, reference_cls = self._run_policy_rollout(
                    self.ref_policy,
                    initial_sample.detach(),
                    ego_query.detach(),
                    agents_query.detach(),
                    bev_feature.detach(),
                    bev_spatial_shape,
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
                reference_idx = reference_cls.argmax(dim=-1)
                batch_idx = torch.arange(
                    reference_reg.shape[0], device=reference_reg.device
                )
                reference_trajectory = reference_reg[
                    batch_idx, reference_idx
                ].unsqueeze(1)

            with torch.no_grad():
                if self._lazy_metric_cache is not None:
                    reference_result = self._compute_rewards_from_lazy_cache(
                        reference_trajectory, tokens_list, 1
                    )
                elif self.metric_cache_loader is not None:
                    reference_result = self._compute_rewards_from_disk(
                        reference_trajectory, tokens_list, 1
                    )
                else:
                    reference_result = None
            if reference_result is None:
                raise RuntimeError("Reference gate requires a PDM metric cache")
            reference_reward = reference_result["raw_rewards"][:, 0]
            reference_valid = reference_result["valid_mask"][:, 0]
        else:
            # The generator is frozen in selector-only stages. Use the
            # fixed-reference selector's mode on the current candidate group.
            reference_idx = final_ref_poses_cls.detach().argmax(dim=-1)
            reference_reward = raw_rewards.gather(
                1, reference_idx.unsqueeze(-1)
            ).squeeze(-1)
            reference_valid = reward_valid_mask.gather(
                1, reference_idx.unsqueeze(-1)
            ).squeeze(-1)

        return reference_reward.detach(), reference_valid.detach()

    def _compute_reference_anchor_rewards(
        self,
        initial_sample,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        tokens_list,
    ):
        """Score one fixed-reference trajectory for every anchor.

        The deterministic reference rollout starts from the same initial
        diffusion state but is computed before, and independently of, the
        stochastic behavior-policy actions. It is therefore a state/anchor
        baseline rather than an action-dependent control variate.
        """
        with torch.no_grad():
            reference_reg, _ = self._run_policy_rollout(
                self.ref_policy,
                initial_sample.detach(),
                ego_query.detach(),
                agents_query.detach(),
                bev_feature.detach(),
                bev_spatial_shape,
                status_encoding.detach(),
                global_img.detach() if global_img is not None else None,
            )
            num_anchors = reference_reg.shape[1]
            if self._lazy_metric_cache is not None:
                reference_result = self._compute_rewards_from_lazy_cache(
                    reference_reg, tokens_list, num_anchors
                )
            elif self.metric_cache_loader is not None:
                reference_result = self._compute_rewards_from_disk(
                    reference_reg, tokens_list, num_anchors
                )
            else:
                reference_result = None
        if reference_result is None:
            raise RuntimeError(
                "Anchor-conditioned reference baseline requires a PDM metric cache"
            )
        return (
            reference_result["raw_rewards"].detach(),
            reference_result["valid_mask"].detach(),
        )

    def _run_selector_policy_with_generation_kl(
        self,
        initial_sample,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    ):
        """Run deployed refinement and constrain both means to frozen base."""
        self.diffusion_scheduler.set_timesteps(
            self._scheduler_num_inference_steps, initial_sample.device
        )
        sample = initial_sample
        step_kls = []
        final_reg = final_cls = None
        sigmas = self.get_generation_trust_sigmas()
        for step_index, timestep in enumerate(self._roll_timesteps):
            current_reg, current_cls = _decode_policy_step(
                self, self.diff_decoder, sample, timestep, ego_query,
                agents_query, bev_feature, bev_spatial_shape, status_encoding,
                global_img,
            )
            with torch.no_grad():
                reference_reg, _ = _decode_policy_step(
                    self, self.ref_policy, sample.detach(), timestep,
                    ego_query.detach(), agents_query.detach(),
                    bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
            if step_index == 0:
                current_mean = self.diffusion_scheduler.step(
                    model_output=self.norm_odo(current_reg[..., :2]),
                    timestep=timestep, sample=sample,
                ).prev_sample
                with torch.no_grad():
                    reference_mean = self.diffusion_scheduler.step(
                        model_output=self.norm_odo(reference_reg[..., :2]),
                        timestep=timestep, sample=sample.detach(),
                    ).prev_sample.detach()
                std = current_mean.new_tensor(sigmas["transition"])
                sample = current_mean
            else:
                current_mean = self.norm_odo(current_reg)
                reference_mean = self.norm_odo(reference_reg).detach()
                std = current_mean.new_tensor(sigmas["final"])
                final_reg = current_reg
                final_cls = current_cls
            step_kls.append(
                diagonal_gaussian_kl_same_std(current_mean, reference_mean, std)
            )
        if final_reg is None or final_cls is None or len(step_kls) != 2:
            raise RuntimeError(
                "selector_group requires the registered [8, 0] schedule"
            )
        return final_reg, final_cls, torch.stack(step_kls, dim=-1)

    def forward_train_grpo(
        self, ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, targets=None, global_img=None, tokens_list=None,
    ) -> Dict[str, torch.Tensor]:
        """Two-step mode-selection rollout with current, old and reference policies."""
        if self.ref_policy is None or self.old_policy is None:
            raise RuntimeError("GRPO requires both reference and old policies")

        self.diff_decoder.eval()
        self.ref_policy.eval()
        self.old_policy.eval()

        bs = ego_query.shape[0]
        device = ego_query.device
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)
        normalized_anchor = self.norm_odo(plan_anchor)
        trunc_timesteps = torch.full(
            (bs,), self._truncation_timestep, device=device, dtype=torch.long
        )
        initial_sample = self.diffusion_scheduler.add_noise(
            original_samples=normalized_anchor,
            noise=torch.randn_like(normalized_anchor),
            timesteps=trunc_timesteps,
        )

        if self._grpo_training_mode == "paired_tail_risk_selector":
            with torch.no_grad():
                current_reg, current_cls = self._run_policy_rollout(
                    self.diff_decoder, initial_sample, ego_query, agents_query,
                    bev_feature, bev_spatial_shape, status_encoding, global_img,
                )
                base_reg, reference_cls = self._run_policy_rollout(
                    self.ref_policy, initial_sample.detach(), ego_query.detach(),
                    agents_query.detach(), bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
            pair_outputs = self.paired_risk_head(
                current_reg.detach(), base_reg.detach(), bev_feature.detach(),
                bev_spatial_shape, agents_query.detach(), ego_query.detach(),
                status_encoding.detach(),
            )
            paired_trajectories = torch.cat(
                (current_reg.detach(), base_reg.detach()), dim=1
            )
            reward_result = None
            if tokens_list is not None:
                if self._lazy_metric_cache is not None:
                    reward_result = self._compute_rewards_from_lazy_cache(
                        paired_trajectories, tokens_list,
                        paired_trajectories.shape[1],
                    )
                elif self.metric_cache_loader is not None:
                    reward_result = self._compute_rewards_from_disk(
                        paired_trajectories, tokens_list,
                        paired_trajectories.shape[1],
                    )
            if reward_result is None:
                raise RuntimeError(
                    "paired-risk training requires PDM rewards and scene tokens"
                )
            modes = current_reg.shape[1]
            self._paired_risk_training_updates.add_(1)
            mode_idx = reference_cls.argmax(dim=-1)
            best_reg = current_reg[torch.arange(bs, device=device), mode_idx]
            return {
                "trajectory": best_reg,
                "final_poses_reg": current_reg,
                "final_poses_cls": current_cls,
                "final_old_poses_cls": current_cls.detach(),
                "final_ref_poses_cls": reference_cls.detach(),
                "rewards": reward_result["training_rewards"][:, :modes],
                "raw_rewards": reward_result["raw_rewards"][:, :modes],
                "reward_valid_mask": reward_result["valid_mask"][:, :modes],
                "component_scores": reward_result["component_scores"][:, :modes],
                "paired_risk_event_logits": pair_outputs["event_logits"],
                "paired_risk_delta_prediction": pair_outputs["delta_prediction"],
                "paired_current_rewards": reward_result["raw_rewards"][:, :modes],
                "paired_base_rewards": reward_result["raw_rewards"][:, modes:],
                "paired_current_components": reward_result["component_scores"][:, :modes],
                "paired_base_components": reward_result["component_scores"][:, modes:],
                "paired_current_valid": reward_result["valid_mask"][:, :modes],
                "paired_base_valid": reward_result["valid_mask"][:, modes:],
                "paired_reference_logits": reference_cls.detach(),
                "grpo_training_rollout": True,
                "num_modes": modes,
                "mode_idx": mode_idx,
            }

        if self._grpo_training_mode in {
            "stage24_selector", "stage25_relative_harm_selector",
            "stage37_jfi_selector",
        }:
            stage25_training = (
                self._grpo_training_mode == "stage25_relative_harm_selector"
            )
            stage37_jfi_training = (
                self._grpo_training_mode == "stage37_jfi_selector"
            )
            if tokens_list is None:
                raise RuntimeError("Stage24 selector training requires scene tokens")
            missing = [
                token for token in tokens_list
                if token not in self._stage24_candidate_bank
            ]
            if missing:
                raise RuntimeError(f"Stage24 candidate bank lacks token: {missing[0]}")
            steps_per_epoch = int(
                getattr(self._config, "stage24_selector_steps_per_epoch", 0)
            )
            training_updates = (
                self._stage37_jfi_training_updates
                if stage37_jfi_training
                else self._stage25_selector_training_updates
                if stage25_training else self._stage24_selector_training_updates
            )
            epoch_index = int(training_updates.item()) // steps_per_epoch
            combination_index = epoch_index % len(self._stage24_bank_combinations)
            bank_records = [
                self._stage24_candidate_bank[token][combination_index]
                for token in tokens_list
            ]
            reference_reg = torch.as_tensor(
                [record["candidate_trajectories"] for record in bank_records],
                device=device, dtype=bev_feature.dtype,
            )
            reference_cls = torch.as_tensor(
                [record["candidate_reference_logits"] for record in bank_records],
                device=device, dtype=bev_feature.dtype,
            )
            raw_reward_rows = [record["candidate_rewards"] for record in bank_records]
            valid = torch.as_tensor(
                [[value is not None for value in row] for row in raw_reward_rows],
                device=device, dtype=torch.bool,
            )
            raw_rewards = torch.as_tensor(
                [
                    [
                        float(value) if value is not None else float("nan")
                        for value in row
                    ]
                    for row in raw_reward_rows
                ],
                device=device, dtype=torch.float32,
            )
            component_scores = torch.as_tensor(
                [record["candidate_components"] for record in bank_records],
                device=device, dtype=torch.float32,
            )
            selector_outputs = self.stage24_selector(
                reference_reg, reference_cls, bev_feature.detach(),
                bev_spatial_shape, agents_query.detach(), ego_query.detach(),
                status_encoding.detach(),
            )
            log_ids = tuple(self._stage24_token_logs[token] for token in tokens_list)
            train_members = stage24_member_training_mask(
                log_ids, self.stage24_selector.num_members,
                float(getattr(
                    self._config, "stage24_selector_subbag_fraction", 0.8
                )),
                reference_reg.device,
            )
            if stage25_training:
                stage25_outputs = self.stage25_selector(
                    selector_outputs["embedding_predictions"].detach()
                )
                self._stage25_selector_training_updates.add_(1)
            elif stage37_jfi_training:
                stage37_jfi_outputs = self.stage37_jfi_selector(
                    selector_outputs["embedding_predictions"].detach()
                )
                self._stage37_jfi_training_updates.add_(1)
            else:
                embedding_rows = selector_outputs[
                    "embedding_mean"
                ].detach().double().reshape(
                    -1, self.stage24_selector.selector_dim
                )
                finite_embedding = torch.isfinite(embedding_rows).all(dim=-1)
                embedding_rows = embedding_rows[finite_embedding]
                self._stage24_embedding_count.add_(embedding_rows.shape[0])
                self._stage24_embedding_sum.add_(embedding_rows.sum(dim=0))
                self._stage24_embedding_sum_sq.add_(
                    embedding_rows.square().sum(dim=0)
                )
                self._stage24_selector_training_updates.add_(1)
            fallback_mode = selector_outputs["fallback_mode"]
            batch_index = torch.arange(reference_reg.shape[0], device=device)
            best_reg = reference_reg[batch_index, fallback_mode]
            safety_weights, any_weight = self._stage24_positive_weights
            result = {
                "trajectory": best_reg,
                "final_poses_reg": reference_reg,
                "final_poses_cls": reference_cls,
                "final_old_poses_cls": reference_cls.detach(),
                "final_ref_poses_cls": reference_cls.detach(),
                "rewards": raw_rewards,
                "raw_rewards": raw_rewards,
                "reward_valid_mask": valid,
                "component_scores": component_scores,
                "stage24_safety_logits": selector_outputs["safety_logits"],
                "stage24_safety_probabilities": selector_outputs[
                    "safety_probabilities"
                ],
                "stage24_any_unsafe_logits": selector_outputs[
                    "any_unsafe_logits"
                ],
                "stage24_any_unsafe_probabilities": selector_outputs[
                    "any_unsafe_probabilities"
                ],
                "stage24_component_delta_predictions": selector_outputs[
                    "component_delta_predictions"
                ],
                "stage24_delta_predictions": selector_outputs[
                    "delta_predictions"
                ],
                "stage24_embedding_predictions": selector_outputs[
                    "embedding_predictions"
                ],
                "stage24_embedding_mean": selector_outputs["embedding_mean"],
                "stage24_member_training_mask": train_members,
                "stage24_fallback_mode": fallback_mode,
                "stage24_safety_positive_weights": torch.tensor(
                    safety_weights, device=device, dtype=torch.float32,
                ),
                "stage24_any_unsafe_positive_weight": torch.tensor(
                    any_weight, device=device, dtype=torch.float32,
                ),
                "stage24_bank_combination_index": torch.tensor(
                    combination_index, device=device, dtype=torch.long,
                ),
                "grpo_training_rollout": True,
                "num_modes": 20,
                "mode_idx": fallback_mode,
            }
            if stage25_training:
                result.update({
                    "stage25_harm_logits": stage25_outputs["harm_logits"],
                    "stage25_catastrophe_logits": stage25_outputs[
                        "catastrophe_logits"
                    ],
                    "stage25_positive_weights": torch.tensor(
                        self._stage25_positive_weights,
                        device=device, dtype=torch.float32,
                    ),
                })
            if stage37_jfi_training:
                result.update({
                    "stage37_jfi_joint_logits": stage37_jfi_outputs[
                        "joint_logits"
                    ],
                    "stage37_jfi_delta_quantiles": stage37_jfi_outputs[
                        "delta_quantiles"
                    ],
                })
            return result
        if self._grpo_training_mode == "stage23_selector":
            if tokens_list is None:
                raise RuntimeError("Stage23 selector training requires scene tokens")
            missing = [token for token in tokens_list if token not in self._stage23_candidate_bank]
            if missing:
                raise RuntimeError(f"Stage23 candidate bank lacks token: {missing[0]}")
            bank_records = [
                record
                for token in tokens_list
                for record in self._stage23_candidate_bank[token]
            ]
            reference_reg = torch.as_tensor(
                [record["candidate_trajectories"] for record in bank_records],
                device=device, dtype=bev_feature.dtype,
            )
            reference_cls = torch.as_tensor(
                [record["candidate_reference_logits"] for record in bank_records],
                device=device, dtype=bev_feature.dtype,
            )
            raw_reward_rows = [record["candidate_rewards"] for record in bank_records]
            valid = torch.as_tensor(
                [[value is not None for value in row] for row in raw_reward_rows],
                device=device, dtype=torch.bool,
            )
            raw_rewards = torch.as_tensor(
                [
                    [float(value) if value is not None else float("nan") for value in row]
                    for row in raw_reward_rows
                ],
                device=device, dtype=torch.float32,
            )
            component_scores = torch.as_tensor(
                [record["candidate_components"] for record in bank_records],
                device=device, dtype=torch.float32,
            )
            namespaces_per_scene = len(self._stage23_candidate_bank[tokens_list[0]])
            repeated_bev = bev_feature.detach().repeat_interleave(
                namespaces_per_scene, dim=0
            )
            repeated_agents = agents_query.detach().repeat_interleave(
                namespaces_per_scene, dim=0
            )
            repeated_ego = ego_query.detach().repeat_interleave(
                namespaces_per_scene, dim=0
            )
            repeated_status = status_encoding.detach().repeat_interleave(
                namespaces_per_scene, dim=0
            )
            selector_outputs = self.stage23_selector(
                reference_reg, reference_cls, repeated_bev, bev_spatial_shape,
                repeated_agents, repeated_ego, repeated_status,
            )
            log_ids = tuple(
                self._stage23_token_logs[token]
                for token in tokens_list
                for _ in range(namespaces_per_scene)
            )
            train_members = member_training_mask(
                log_ids, self.stage23_selector.num_members, reference_reg.device
            )
            self._stage23_selector_training_updates.add_(1)
            fallback_mode = reference_cls.argmax(dim=-1)
            bank_batch = reference_reg.shape[0]
            best_reg = reference_reg[
                torch.arange(bank_batch, device=device), fallback_mode
            ]
            return {
                "trajectory": best_reg,
                "final_poses_reg": reference_reg,
                "final_poses_cls": reference_cls,
                "final_old_poses_cls": reference_cls.detach(),
                "final_ref_poses_cls": reference_cls.detach(),
                "rewards": raw_rewards,
                "raw_rewards": raw_rewards,
                "reward_valid_mask": valid,
                "component_scores": component_scores,
                "stage23_component_logits": selector_outputs["component_logits"],
                "stage23_component_predictions": selector_outputs["component_predictions"],
                "stage23_score_predictions": selector_outputs["score_predictions"],
                "stage23_member_training_mask": train_members,
                "stage23_log_buckets": (~train_members).to(torch.long).argmax(dim=-1),
                "grpo_training_rollout": True,
                "num_modes": 20,
                "mode_idx": fallback_mode,
            }

        if self._grpo_training_mode == "value_selector":
            with torch.no_grad():
                current_reg, _ = self._run_policy_rollout(
                    self.diff_decoder, initial_sample, ego_query, agents_query,
                    bev_feature, bev_spatial_shape, status_encoding, global_img,
                )
                reference_reg, reference_cls = self._run_policy_rollout(
                    self.ref_policy, initial_sample.detach(), ego_query.detach(),
                    agents_query.detach(), bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
                value_trajectories = torch.cat(
                    [current_reg.detach(), reference_reg.detach()], dim=1
                )
                value_reference_logits = torch.cat(
                    [reference_cls.detach(), reference_cls.detach()], dim=1
                )
            value_outputs = self.value_selector(
                value_trajectories,
                bev_feature.detach(),
                bev_spatial_shape,
                agents_query.detach(),
                ego_query.detach(),
                status_encoding.detach(),
            )
            reward_result = None
            if tokens_list is not None:
                if self._lazy_metric_cache is not None:
                    reward_result = self._compute_rewards_from_lazy_cache(
                        value_trajectories, tokens_list, value_trajectories.shape[1]
                    )
                elif self.metric_cache_loader is not None:
                    reward_result = self._compute_rewards_from_disk(
                        value_trajectories, tokens_list, value_trajectories.shape[1]
                    )
            if reward_result is None:
                raise RuntimeError(
                    "value-selector training requires PDM rewards and scene tokens"
                )
            bootstrap = deterministic_bootstrap_mask(
                tokens_list,
                self.value_selector.num_heads,
                value_trajectories.device,
                float(getattr(self._config, "value_selector_bootstrap_fraction", 0.8)),
            )
            self._value_selector_training_updates.add_(1)
            fallback_mode = reference_cls.argmax(dim=-1)
            best_reg = current_reg[
                torch.arange(bs, device=device), fallback_mode
            ]
            return {
                "trajectory": best_reg,
                "final_poses_reg": value_trajectories,
                "final_poses_cls": value_reference_logits,
                "final_old_poses_cls": value_reference_logits,
                "final_ref_poses_cls": value_reference_logits,
                "rewards": reward_result["training_rewards"],
                "raw_rewards": reward_result["raw_rewards"],
                "reward_valid_mask": reward_result["valid_mask"],
                "component_scores": reward_result["component_scores"],
                "value_component_predictions": value_outputs[
                    "component_predictions"
                ],
                "value_score_predictions": value_outputs["score_predictions"],
                "value_bootstrap_mask": bootstrap,
                "value_reference_logits": value_reference_logits,
                "value_group_size": current_reg.shape[1],
                "grpo_training_rollout": True,
                "num_modes": value_trajectories.shape[1],
                "mode_idx": fallback_mode,
            }

        generation_outputs = {}
        paired_base_poses_reg = None
        stage39_deferred_trace = None
        stage36_deferred_trace = None
        stage37_deferred_trace = None
        stage38_deferred_trace = None
        if self._grpo_training_mode == "diffgrpo_full_chain":
            trace = collect_full_chain_diffgrpo_trace(
                self,
                initial_sample,
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
            )
            final_poses_reg = trace["trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
            }
        elif self._grpo_training_mode in {
            "diffgrpo_selected_anchor",
            "diffgrpo_selected_anchor_base_preserve",
        }:
            selected_modes = self._lookup_selected_anchor_modes(
                tokens_list, device
            )
            clean_anchors = self.plan_anchor.unsqueeze(0).expand(
                bs, -1, -1, -1
            )
            gather_index = selected_modes[:, None, None, None].expand(
                -1, 1, clean_anchors.shape[2], clean_anchors.shape[3]
            )
            selected_clean_sample = self.norm_odo(
                clean_anchors.gather(1, gather_index)
            )
            trace = collect_selected_anchor_diffgrpo_trace(
                self,
                selected_clean_sample,
                selected_modes,
                self._diffgrpo_group_size,
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
            )
            final_poses_reg = trace["trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_num_denoising_steps": trace[
                    "num_denoising_steps"
                ],
                "diffgrpo_selected_anchor_modes": trace[
                    "selected_anchor_modes"
                ],
                "diffgrpo_group_size": trace["group_size"],
            }
            if self._grpo_training_mode == "diffgrpo_selected_anchor_base_preserve":
                metadata = self._lookup_base_preserve_metadata(tokens_list, device)
                generation_outputs.update(
                    {
                        "diffgrpo_base_rewards": metadata["base_rewards"],
                        "diffgrpo_base_safety": metadata["base_safety"],
                        "diffgrpo_scene_weights": metadata["scene_weights"],
                        "diffgrpo_bc_weights": metadata["bc_weights"],
                    }
                )
        elif self._grpo_training_mode == "diffgrpo_paired_residual":
            selected_modes = self._lookup_selected_anchor_modes(
                tokens_list, device
            )
            clean_anchors = self.plan_anchor.unsqueeze(0).expand(
                bs, -1, -1, -1
            )
            gather_index = selected_modes[:, None, None, None].expand(
                -1, 1, clean_anchors.shape[2], clean_anchors.shape[3]
            )
            selected_clean_sample = self.norm_odo(
                clean_anchors.gather(1, gather_index)
            )
            trace = collect_paired_residual_diffgrpo_trace(
                self,
                selected_clean_sample,
                selected_modes,
                self._diffgrpo_group_size,
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_selected_anchor_modes": trace[
                    "selected_anchor_modes"
                ],
                "diffgrpo_group_size": trace["group_size"],
                "diffgrpo_scene_weights": self._lookup_paired_residual_weights(
                    tokens_list, device
                ),
            }
        elif self._grpo_training_mode in STAGE39_TRAINING_MODES:
            trace = collect_stage39_sampling_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            stage39_deferred_trace = trace
            final_poses_reg = self.denorm_odo(trace["challenger_final"])
            paired_base_poses_reg = self.denorm_odo(trace["public_final"])
            final_poses_cls = trace["challenger_cls"]
            final_old_poses_cls = final_poses_cls.detach()
            final_ref_poses_cls = trace["public_cls"].detach()
            generation_outputs = {
                "diffgrpo_group_size": trace["group_size"],
                "stage39_sampled_chain_count": trace["sampled_chain_count"],
                "stage39_independent_initial_noise": trace[
                    "independent_initial_noise"
                ],
                "stage39_initial_noise_abs_cosine": trace[
                    "initial_noise_abs_cosine"
                ],
            }
        elif self._grpo_training_mode in STAGE38_ESCR_TRAINING_MODES:
            trace = collect_stage38_sampling_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            stage38_deferred_trace = trace
            final_poses_reg = self.denorm_odo(trace["current_final"])
            paired_base_poses_reg = self.denorm_odo(trace["public_final"])
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_group_size": trace["group_size"],
                "stage38_current_selected_modes": trace[
                    "current_selected_modes"
                ],
                "stage38_scene_buckets": self._lookup_stage30_buckets(
                    tokens_list, device
                ),
                "stage38_selector_mode_disagreement": trace[
                    "selector_mode_disagreement"
                ],
                "stage38_current_selector_switch_rate": trace[
                    "current_selector_switch_rate"
                ],
                "stage38_public_selector_switch_rate": trace[
                    "public_selector_switch_rate"
                ],
                "stage38_sampled_chain_count": trace[
                    "sampled_chain_count"
                ],
                "stage38_counterfactual_selector_count": trace[
                    "counterfactual_selector_count"
                ],
            }
        elif self._grpo_training_mode in STAGE37_BPD_TRAINING_MODES:
            trace = collect_stage37_sampling_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            stage37_deferred_trace = trace
            final_poses_reg = self.denorm_odo(trace["current_final"])
            paired_base_poses_reg = self.denorm_odo(trace["public_final"])
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_group_size": trace["group_size"],
                "stage37_current_selected_modes": trace["current_selected_modes"],
                "stage37_public_selected_modes": trace["public_selected_modes"],
                "stage37_active_modes": trace["active_modes"],
                "stage37_active_valid": trace["active_valid"],
                "stage37_hybrid_selected_modes": trace["hybrid_selected_modes"],
                "stage37_donor_indices": trace["donor_indices"],
                "stage37_scene_buckets": self._lookup_stage30_buckets(
                    tokens_list, device
                ),
                "stage37_selector_mode_disagreement": trace[
                    "selector_mode_disagreement"
                ],
                "stage37_current_selector_switch_rate": trace[
                    "current_selector_switch_rate"
                ],
                "stage37_public_selector_switch_rate": trace[
                    "public_selector_switch_rate"
                ],
                "stage37_sampled_chain_count": trace["sampled_chain_count"],
                "stage37_counterfactual_selector_count": trace[
                    "counterfactual_selector_count"
                ],
            }
        elif self._grpo_training_mode in STAGE36_RGT_NCD_TRAINING_MODES:
            trace = collect_stage36_sampling_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            stage36_deferred_trace = trace
            final_poses_reg = self.denorm_odo(trace["current_final"])
            paired_base_poses_reg = self.denorm_odo(trace["public_final"])
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_group_size": trace["group_size"],
                "stage36_current_selected_modes": trace["current_selected_modes"],
                "stage36_public_selected_modes": trace["public_selected_modes"],
                "stage36_active_modes": trace["active_modes"],
                "stage36_active_valid": trace["active_valid"],
                "stage36_hybrid_selected_modes": trace["hybrid_selected_modes"],
                "stage36_donor_indices": trace["donor_indices"],
                "stage36_scene_buckets": self._lookup_stage30_buckets(
                    tokens_list, device
                ),
                "stage36_selector_mode_disagreement": trace[
                    "selector_mode_disagreement"
                ],
                "stage36_current_selector_switch_rate": trace[
                    "current_selector_switch_rate"
                ],
                "stage36_public_selector_switch_rate": trace[
                    "public_selector_switch_rate"
                ],
                "stage36_sampled_chain_count": trace["sampled_chain_count"],
                "stage36_counterfactual_selector_count": trace[
                    "counterfactual_selector_count"
                ],
            }
        elif self._grpo_training_mode in DEPLOYED_SET_TRAINING_MODES:
            trace = collect_deployed_selected_set_diffgrpo_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_group_size": trace["group_size"],
                "stage31_current_selected_modes": trace[
                    "current_selected_modes"
                ],
                "stage31_public_selected_modes": trace[
                    "public_selected_modes"
                ],
                "stage31_selector_mode_disagreement": trace[
                    "selector_mode_disagreement"
                ],
                "stage31_current_selector_switch_rate": trace[
                    "current_selector_switch_rate"
                ],
                "stage31_public_selector_switch_rate": trace[
                    "public_selector_switch_rate"
                ],
                "stage31_current_sampled_chain_count": trace[
                    "current_sampled_chain_count"
                ],
                "stage31_public_sampled_chain_count": trace[
                    "public_sampled_chain_count"
                ],
                "stage31_current_replayed_chain_count": trace[
                    "current_replayed_chain_count"
                ],
                "stage31_public_replayed_chain_count": trace[
                    "public_replayed_chain_count"
                ],
            }
        elif self._grpo_training_mode in STAGE32_FRONTIER_TRAINING_MODES:
            trace = collect_selector_aware_frontier_diffgrpo_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_group_size": trace["group_size"],
                "stage32_current_selected_modes": trace[
                    "current_selected_modes"
                ],
                "stage32_public_selected_modes": trace[
                    "public_selected_modes"
                ],
                "stage32_current_mode_valid": trace["current_mode_valid"].reshape(bs, -1),
                "stage32_public_mode_valid": trace["public_mode_valid"].reshape(bs, -1),
                "stage32_pool_size": trace["stage32_pool_size"],
                "stage32_selector_mode_disagreement": trace[
                    "selector_mode_disagreement"
                ],
                "stage32_current_selector_switch_rate": trace[
                    "current_selector_switch_rate"
                ],
                "stage32_public_selector_switch_rate": trace[
                    "public_selector_switch_rate"
                ],
                "stage32_current_sampled_chain_count": trace[
                    "current_sampled_chain_count"
                ],
                "stage32_public_sampled_chain_count": trace[
                    "public_sampled_chain_count"
                ],
                "stage32_current_replayed_chain_count": trace[
                    "current_replayed_chain_count"
                ],
                "stage32_public_replayed_chain_count": trace[
                    "public_replayed_chain_count"
                ],
            }
        elif self._grpo_training_mode in STAGE35_NCD_TRAINING_MODES:
            trace = collect_nested_counterfactual_deployment_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_group_size": trace["group_size"],
                "stage35_current_selected_modes": trace["current_selected_modes"],
                "stage35_public_selected_modes": trace["public_selected_modes"],
                "stage35_active_modes": trace["active_modes"],
                "stage35_active_valid": trace["active_valid"],
                "stage35_hybrid_selected_modes": trace["hybrid_selected_modes"],
                "stage35_donor_indices": trace["donor_indices"],
                "stage35_scene_buckets": self._lookup_stage30_buckets(tokens_list, device),
                "stage35_selector_mode_disagreement": trace["selector_mode_disagreement"],
                "stage35_current_selector_switch_rate": trace["current_selector_switch_rate"],
                "stage35_public_selector_switch_rate": trace["public_selector_switch_rate"],
                "stage35_sampled_chain_count": trace["sampled_chain_count"],
                "stage35_replayed_chain_count": trace["replayed_chain_count"],
                "stage35_counterfactual_selector_count": trace["counterfactual_selector_count"],
            }
        elif self._grpo_training_mode in STAGE34_MODE_ALIGNED_TRAINING_MODES:
            trace = collect_mode_coverage_diffgrpo_trace(
                self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_group_size": trace["group_size"],
                "stage34_sampled_chain_count": trace["sampled_chain_count"],
                "stage34_replayed_chain_count": trace["replayed_chain_count"],
                "stage34_scene_buckets": self._lookup_stage30_buckets(tokens_list, device),
            }
            generation_outputs.update(build_stage34_trace_diagnostics(
                self, trace, ego_query, agents_query, bev_feature,
                bev_spatial_shape, status_encoding,
            ))
        elif self._grpo_training_mode in MODE_COVERAGE_TRAINING_MODES:
            trace = collect_mode_coverage_diffgrpo_trace(
                self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_group_size": trace["group_size"],
                "stage30_sampled_chain_count": trace["sampled_chain_count"],
                "stage30_replayed_chain_count": trace["replayed_chain_count"],
                "stage30_scene_buckets": self._lookup_stage30_buckets(
                    tokens_list, device
                ),
            }
        elif self._grpo_training_mode in SELECTED_SET_TRAINING_MODES:
            trace = collect_selected_set_diffgrpo_trace(
                self, self._diffgrpo_group_size, ego_query, agents_query,
                bev_feature, bev_spatial_shape, status_encoding, global_img,
            )
            final_poses_reg = trace["trajectories"]
            paired_base_poses_reg = trace["base_trajectories"]
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["current_cls"].detach()
            final_ref_poses_cls = trace["reference_cls"].detach()
            generation_outputs = {
                "diffgrpo_current_log_probs": trace["current_log_probs"],
                "diffgrpo_bc_log_probs": trace["bc_log_probs"],
                "diffgrpo_reference_mean_kl": trace["reference_mean_kl"],
                "diffgrpo_num_denoising_steps": trace["num_denoising_steps"],
                "diffgrpo_selected_set_modes": trace["selected_modes"],
                "diffgrpo_group_size": trace["group_size"],
                "stage23_selector_switch_rate": trace["selector_switch_rate"],
                "stage23_sampled_chain_count": trace["sampled_chain_count"],
                "stage23_replayed_chain_count": trace["replayed_chain_count"],
            }
        elif self._grpo_training_mode in {
            "generation", "generation_group", "generation_group_adaptive", "joint"
        }:
            rollout_count = self._grpo_rollouts_per_mode
            if rollout_count == 1:
                trace_initial_sample = initial_sample
                trace_ego_query = ego_query
                trace_agents_query = agents_query
                trace_bev_feature = bev_feature
                trace_status_encoding = status_encoding
                trace_global_img = global_img
            else:
                trace_initial_sample = initial_sample.repeat_interleave(
                    rollout_count, dim=0
                )
                trace_ego_query = ego_query.repeat_interleave(rollout_count, dim=0)
                trace_agents_query = agents_query.repeat_interleave(
                    rollout_count, dim=0
                )
                trace_bev_feature = bev_feature.repeat_interleave(
                    rollout_count, dim=0
                )
                trace_status_encoding = status_encoding.repeat_interleave(
                    rollout_count, dim=0
                )
                trace_global_img = (
                    global_img.repeat_interleave(rollout_count, dim=0)
                    if global_img is not None
                    else None
                )
            trace = collect_generation_trace(
                self,
                trace_initial_sample,
                trace_ego_query,
                trace_agents_query,
                trace_bev_feature,
                bev_spatial_shape,
                trace_status_encoding,
                trace_global_img,
            )
            trace = {
                key: flatten_generation_rollouts(value, bs, rollout_count)
                for key, value in trace.items()
            }
            final_poses_reg = trace["trajectories"]
            # Always expose current selector logits. Generation-only keeps the
            # classification branches frozen and its selection policy weight
            # at zero; the logits support metrics and optional consistency KL.
            final_poses_cls = trace["current_cls"]
            final_old_poses_cls = trace["old_cls"]
            final_ref_poses_cls = trace["reference_cls"]
            trust_metrics = {}
            if "trust_pre_distance" in trace:
                trust_metrics = summarize_trust_projection(
                    trace["trust_pre_distance"],
                    trace["trust_post_distance"],
                    trace["trust_alpha"],
                    trace["trust_projected"],
                    trace["trust_reference_coverage"],
                )
            generation_outputs = {
                "generation_current_log_probs": trace["current_log_probs"],
                "generation_old_log_probs": trace["old_log_probs"],
                "generation_reference_log_probs": trace["reference_log_probs"],
                "generation_kl": trace["generation_kl"],
                **trust_metrics,
            }
        elif self._grpo_training_mode == "selector_group":
            final_poses_reg, final_poses_cls, selector_generation_kl = (
                self._run_selector_policy_with_generation_kl(
                    initial_sample, ego_query, agents_query, bev_feature,
                    bev_spatial_shape, status_encoding, global_img,
                )
            )
            generation_outputs = {
                "selector_generation_kl": selector_generation_kl,
            }
            with torch.no_grad():
                _, final_old_poses_cls = self._run_policy_rollout(
                    self.old_policy, initial_sample.detach(), ego_query.detach(),
                    agents_query.detach(), bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
                _, final_ref_poses_cls = self._run_policy_rollout(
                    self.ref_policy, initial_sample.detach(), ego_query.detach(),
                    agents_query.detach(), bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
        else:
            final_poses_reg, final_poses_cls = self._run_policy_rollout(
                self.diff_decoder, initial_sample, ego_query, agents_query, bev_feature,
                bev_spatial_shape, status_encoding, global_img,
            )
            with torch.no_grad():
                _, final_old_poses_cls = self._run_policy_rollout(
                    self.old_policy, initial_sample.detach(), ego_query.detach(),
                    agents_query.detach(), bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(), global_img.detach() if global_img is not None else None,
                )
                _, final_ref_poses_cls = self._run_policy_rollout(
                    self.ref_policy, initial_sample.detach(), ego_query.detach(),
                    agents_query.detach(), bev_feature.detach(), bev_spatial_shape,
                    status_encoding.detach(), global_img.detach() if global_img is not None else None,
                )

        rewards = None
        raw_rewards = None
        reward_valid_mask = None
        reward_tiebreak_epsilon = None
        component_scores = None
        num_modes = final_poses_cls.shape[-1]
        score_poses_reg = final_poses_reg
        score_num_modes = num_modes
        if self._grpo_training_mode in {
            "diffgrpo_paired_residual", "diffgrpo_selected_set",
            "stage27_public_diffgrpo_selected_set",
            "stage28_public_paired_uplift_multi",
            "stage28_public_paired_uplift_explore",
            "stage29_public_headroom_hybrid",
            "stage29_public_headroom_conditional",
            "stage30_public_mode_coverage",
            "stage30_public_mode_coverage_constrained",
            "stage31_public_deployed_pair",
            "stage31_public_deployed_frontier",
            "stage32_public_deployed_extended",
            "stage32_selector_aware_frontier",
            "stage33_cdc_grpo",
            "stage34_mode_aligned_frontier_grpo",
            "stage35_nested_counterfactual_deployment_grpo",
            "stage36_reference_gated_tail_ncd_grpo",
            "stage37_bistate_projected_deployment_grpo",
            "stage38_elite_set_counterfactual_repair_grpo",
            "stage39_challenger_bc",
            "stage39_challenger_standard_grpo",
            "stage39_challenger_set_grpo",
        }:
            if paired_base_poses_reg is None:
                raise RuntimeError("paired DiffGRPO trace produced no base trajectories")
            score_poses_reg = torch.cat(
                (final_poses_reg, paired_base_poses_reg), dim=1
            )
            score_num_modes = 2 * num_modes
        if tokens_list is not None:
            if self._lazy_metric_cache is not None:
                reward_result = self._compute_rewards_from_lazy_cache(
                    score_poses_reg, tokens_list, score_num_modes,
                )
            elif self.metric_cache_loader is not None:
                reward_result = self._compute_rewards_from_disk(
                    score_poses_reg, tokens_list, score_num_modes,
                )
            else:
                reward_result = None
            if reward_result is not None:
                if self._grpo_training_mode in {
                    "diffgrpo_paired_residual", "diffgrpo_selected_set",
                    "stage27_public_diffgrpo_selected_set",
                    "stage28_public_paired_uplift_multi",
                    "stage28_public_paired_uplift_explore",
                    "stage29_public_headroom_hybrid",
                    "stage29_public_headroom_conditional",
                    "stage30_public_mode_coverage",
                    "stage30_public_mode_coverage_constrained",
                    "stage31_public_deployed_pair",
                    "stage31_public_deployed_frontier",
                    "stage32_public_deployed_extended",
                    "stage32_selector_aware_frontier",
                    "stage33_cdc_grpo",
                    "stage34_mode_aligned_frontier_grpo",
                    "stage35_nested_counterfactual_deployment_grpo",
                    "stage36_reference_gated_tail_ncd_grpo",
                    "stage39_challenger_bc",
                    "stage39_challenger_standard_grpo",
                    "stage39_challenger_set_grpo",
                    "stage37_bistate_projected_deployment_grpo",
                    "stage38_elite_set_counterfactual_repair_grpo",
                }:
                    rewards = reward_result["training_rewards"][:, :num_modes]
                    raw_rewards = reward_result["raw_rewards"][:, :num_modes]
                    reward_valid_mask = reward_result["valid_mask"][:, :num_modes]
                    # Tie-break epsilon is scene-level [B], not mode-level.
                    # Stage22 optimizes raw PDMS, but retains this diagnostic.
                    reward_tiebreak_epsilon = reward_result[
                        "tiebreak_epsilon"
                    ]
                    component_scores = reward_result[
                        "component_scores"
                    ][:, :num_modes]
                    generation_outputs.update(
                        {
                            "diffgrpo_base_rewards": reward_result[
                                "raw_rewards"
                            ][:, num_modes:],
                            "diffgrpo_base_valid_mask": reward_result[
                                "valid_mask"
                            ][:, num_modes:],
                            "diffgrpo_base_component_scores": reward_result[
                                "component_scores"
                            ][:, num_modes:],
                        }
                    )
                else:
                    rewards = reward_result["training_rewards"]
                    raw_rewards = reward_result["raw_rewards"]
                    reward_valid_mask = reward_result["valid_mask"]
                    reward_tiebreak_epsilon = reward_result["tiebreak_epsilon"]
                    component_scores = reward_result["component_scores"]
        if rewards is None or reward_valid_mask is None:
            raise RuntimeError(
                "GRPO training requires PDM rewards and scene tokens for every batch"
            )
        if not reward_valid_mask.any():
            raise RuntimeError(
                f"No valid PDM reward in batch; tokens={list(tokens_list or [])}"
            )
        if stage36_deferred_trace is not None:
            generation_outputs.update(finalize_stage36_reward_dependent_replay(
                self,
                stage36_deferred_trace,
                rewards=raw_rewards,
                valid_mask=reward_valid_mask,
                component_scores=component_scores,
                base_rewards=generation_outputs["diffgrpo_base_rewards"],
                base_valid_mask=generation_outputs["diffgrpo_base_valid_mask"],
                base_component_scores=generation_outputs[
                    "diffgrpo_base_component_scores"
                ],
                scene_buckets=generation_outputs["stage36_scene_buckets"],
                ego_query=ego_query,
                agents_query=agents_query,
                bev_feature=bev_feature,
                bev_spatial_shape=bev_spatial_shape,
                status_encoding=status_encoding,
                global_img=global_img,
            ))
        if stage37_deferred_trace is not None:
            generation_outputs.update(finalize_stage37_reward_dependent_replay(
                self,
                stage37_deferred_trace,
                rewards=raw_rewards,
                valid_mask=reward_valid_mask,
                component_scores=component_scores,
                base_rewards=generation_outputs["diffgrpo_base_rewards"],
                base_valid_mask=generation_outputs["diffgrpo_base_valid_mask"],
                base_component_scores=generation_outputs[
                    "diffgrpo_base_component_scores"
                ],
                scene_buckets=generation_outputs["stage37_scene_buckets"],
                ego_query=ego_query,
                agents_query=agents_query,
                bev_feature=bev_feature,
                bev_spatial_shape=bev_spatial_shape,
                status_encoding=status_encoding,
                global_img=global_img,
            ))
        if stage38_deferred_trace is not None:
            generation_outputs.update(finalize_stage38_reward_dependent_replay(
                self,
                stage38_deferred_trace,
                rewards=raw_rewards,
                valid_mask=reward_valid_mask,
                component_scores=component_scores,
                base_rewards=generation_outputs["diffgrpo_base_rewards"],
                base_valid_mask=generation_outputs["diffgrpo_base_valid_mask"],
                base_component_scores=generation_outputs[
                    "diffgrpo_base_component_scores"
                ],
                scene_buckets=generation_outputs["stage38_scene_buckets"],
                ego_query=ego_query,
                agents_query=agents_query,
                bev_feature=bev_feature,
                bev_spatial_shape=bev_spatial_shape,
                status_encoding=status_encoding,
                global_img=global_img,
            ))

        if stage39_deferred_trace is not None:
            generation_outputs.update(finalize_stage39_replay(
                self,
                stage39_deferred_trace,
                ego_query=ego_query,
                agents_query=agents_query,
                bev_feature=bev_feature,
                bev_spatial_shape=bev_spatial_shape,
                status_encoding=status_encoding,
                global_img=global_img,
            ))
        reference_selected_reward = None
        reference_reward_valid_mask = None
        reference_anchor_rewards = None
        reference_anchor_valid_mask = None
        if (
            self._grpo_scene_weight_mode == "reference_headroom"
            or self._generation_advantage_mode
            in {"reference_centered", "hierarchical"}
        ):
            reference_selected_reward, reference_reward_valid_mask = (
                self._compute_reference_selected_reward(
                    final_poses_reg=final_poses_reg,
                    final_ref_poses_cls=final_ref_poses_cls,
                    raw_rewards=raw_rewards,
                    reward_valid_mask=reward_valid_mask,
                    initial_sample=initial_sample,
                    ego_query=ego_query,
                    agents_query=agents_query,
                    bev_feature=bev_feature,
                    bev_spatial_shape=bev_spatial_shape,
                    status_encoding=status_encoding,
                    global_img=global_img,
                    tokens_list=tokens_list,
                )
            )

        if self._generation_advantage_mode in {"anchor_hierarchical", "anchor_rloo"}:
            (
                reference_anchor_rewards,
                reference_anchor_valid_mask,
            ) = self._compute_reference_anchor_rewards(
                initial_sample=initial_sample,
                ego_query=ego_query,
                agents_query=agents_query,
                bev_feature=bev_feature,
                bev_spatial_shape=bev_spatial_shape,
                status_encoding=status_encoding,
                global_img=global_img,
                tokens_list=tokens_list,
            )

        if self._grpo_training_mode == "diffgrpo_full_chain":
            mode_idx = final_ref_poses_cls.argmax(dim=-1)
        elif self._grpo_training_mode in {
            "diffgrpo_selected_anchor",
            "diffgrpo_selected_anchor_base_preserve",
            "diffgrpo_paired_residual",
            "diffgrpo_selected_set",
            "stage27_public_diffgrpo_selected_set",
            "stage28_public_paired_uplift_multi",
            "stage28_public_paired_uplift_explore",
            "stage29_public_headroom_hybrid",
            "stage29_public_headroom_conditional",
            "stage30_public_mode_coverage",
            "stage30_public_mode_coverage_constrained",
            "stage31_public_deployed_pair",
            "stage31_public_deployed_frontier",
            "stage32_public_deployed_extended",
            "stage32_selector_aware_frontier",
            "stage33_cdc_grpo",
            "stage39_challenger_bc",
            "stage39_challenger_standard_grpo",
            "stage39_challenger_set_grpo",
            "stage34_mode_aligned_frontier_grpo",
            "stage35_nested_counterfactual_deployment_grpo",
            "stage36_reference_gated_tail_ncd_grpo",
            "stage37_bistate_projected_deployment_grpo",
            "stage38_elite_set_counterfactual_repair_grpo",
        }:
            mode_idx = torch.zeros(bs, dtype=torch.long, device=device)
        else:
            mode_idx = final_poses_cls.argmax(dim=-1)
        best_reg = final_poses_reg[torch.arange(bs, device=device), mode_idx]
        return {
            "trajectory": best_reg,
            "final_poses_reg": final_poses_reg,
            "final_poses_cls": final_poses_cls,
            "final_old_poses_cls": final_old_poses_cls,
            "final_ref_poses_cls": final_ref_poses_cls,
            "rewards": rewards,
            "raw_rewards": raw_rewards,
            "reward_valid_mask": reward_valid_mask,
            "reward_tiebreak_epsilon": reward_tiebreak_epsilon,
            "component_scores": component_scores,
            "reference_selected_reward": reference_selected_reward,
            "reference_reward_valid_mask": reference_reward_valid_mask,
            "reference_anchor_rewards": reference_anchor_rewards,
            "reference_anchor_valid_mask": reference_anchor_valid_mask,
            "grpo_training_rollout": True,
            "num_modes": num_modes,
            "mode_idx": mode_idx,
            **generation_outputs,
        }

    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None, tokens_list=None) -> Dict[str, torch.Tensor]:

        bs = ego_query.shape[0]
        device = ego_query.device

        #plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        #bs, num_mode, ts, d = plan_anchor.shape
        #target_traj = targets["trajectory"]
        #dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1)
        #dist = dist.mean(dim=-1)
        #mode_idx = torch.argmin(dist, dim=-1)
        #noisy_traj_points = plan_anchor

        #timesteps = torch.randint(0, 50, (bs,), device=device)
        
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        bs, num_mode, ts, d = plan_anchor.shape
        target_traj = targets["trajectory"]
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)

        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(0, 50, (bs,), device=device)
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
           original_samples=odo_info_fut,
           noise=noise,
           timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs, 1, -1)

        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img,
        )

        final_poses_reg = poses_reg_list[-1]
        final_poses_cls = poses_cls_list[-1]
        num_modes = final_poses_cls.shape[1]

        rewards = None
        final_ref_poses_cls = None
        kl_div = None

        if tokens_list is not None and self.ref_policy is not None:
            with torch.no_grad():
                ref_poses_reg_list, ref_poses_cls_list = self.ref_policy(
                    traj_feature.detach(),
                    noisy_traj_points,
                    bev_feature.detach(),
                    bev_spatial_shape,
                    agents_query.detach(),
                    ego_query.detach(),
                    time_embed.detach(),
                    status_encoding.detach(),
                    global_img.detach() if global_img is not None else None,
                )
                final_ref_poses_cls = ref_poses_cls_list[-1]

            kl_div = F.kl_div(
                F.log_softmax(final_poses_cls, dim=-1),
                F.softmax(final_ref_poses_cls, dim=-1),
                reduction="batchmean",
            )

            self._reward_step_counter += 1
            should_compute = (
                self._cached_rewards is None
                or self._reward_step_counter % self._reward_compute_interval == 0
                or self._cached_rewards.shape[0] != bs
            )

            if should_compute:
                if self._lazy_metric_cache is not None:
                    rewards = self._compute_rewards_from_lazy_cache(
                        final_poses_reg, tokens_list, num_modes,
                    )
                elif self.metric_cache_loader is not None:
                    rewards = self._compute_rewards_from_disk(
                        final_poses_reg, tokens_list, num_modes,
                    )
                self._cached_rewards = rewards
            else:
                rewards = self._cached_rewards

        best_reg = poses_reg_list[-1][torch.arange(bs), mode_idx]

        return {
            "trajectory": best_reg,
            "final_poses_cls": final_poses_cls,
            "final_ref_poses_cls": final_ref_poses_cls,
            "rewards": rewards,
            "kl_div": kl_div,
            "num_modes": num_modes,
            "mode_idx": mode_idx,
        }

    def _sample_evaluation_noise(
        self, template: torch.Tensor, tokens_list, seed_namespace: int = None
    ) -> torch.Tensor:
        """Generate order-independent noise from scenario tokens during evaluation."""
        if tokens_list is None or len(tokens_list) != template.shape[0]:
            return torch.randn_like(template)

        samples = []
        for token in tokens_list:
            identity = (
                str(token)
                if seed_namespace is None
                else f"{seed_namespace}:{token}"
            )
            digest = hashlib.sha256(identity.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="little") % (2**63 - 1)
            generator = torch.Generator(device=template.device)
            generator.manual_seed(seed)
            samples.append(
                torch.randn(template.shape[1:], device=template.device, dtype=template.dtype, generator=generator)
            )
        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def _forward_test_stage39_candidates(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        tokens_list=None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate Stage39's extra candidate bank without changing public20.

        ``public20`` remains on the legacy path bit-for-bit.  The two explicit
        alternatives are deliberately isolated: ``public40_extra`` adds a
        second deterministic noise bank to the frozen public decoder, while
        ``challenger_union`` replaces that second bank with the independently
        trained challenger decoder.  Both use the ordinary current-logit
        selector so the generator and selector effects remain separable.
        """
        source = str(getattr(self._config, "stage39_candidate_source", "public20"))
        if source not in {"public40_extra", "challenger_union"}:
            raise ValueError(f"unsupported Stage39 candidate source: {source!r}")
        if self.stage39_challenger_decoder is None:
            raise RuntimeError("Stage39 candidate evaluation requires challenger decoder")
        if self._inference_selector_source != "current":
            raise RuntimeError(
                "Stage39 candidate-bank evaluation currently requires selector source=current; "
                "Stage40 will add the learned hierarchical selector"
            )

        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(
            self._scheduler_num_inference_steps, device
        )
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        normalized_anchor = self.norm_odo(plan_anchor)
        if self._evaluation_noise_namespace >= 0:
            namespace = self._evaluation_noise_namespace
        else:
            namespace = None
        public_noise = self._sample_evaluation_noise(
            normalized_anchor, tokens_list, namespace
        )
        trunc_timesteps = torch.full(
            (bs,), self._truncation_timestep, device=device, dtype=torch.long
        )
        public_initial = self.diffusion_scheduler.add_noise(
            original_samples=normalized_anchor,
            noise=public_noise,
            timesteps=trunc_timesteps,
        )
        public_reg, public_cls = self._run_policy_rollout(
            self.diff_decoder, public_initial, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        offset = int(getattr(self._config, "stage39_challenger_noise_offset", 390001))
        extra_namespace = offset if namespace is None else namespace + offset
        extra_noise = self._sample_evaluation_noise(
            normalized_anchor, tokens_list, extra_namespace
        )
        extra_initial = self.diffusion_scheduler.add_noise(
            original_samples=normalized_anchor,
            noise=extra_noise,
            timesteps=trunc_timesteps,
        )
        second_policy = (
            self.diff_decoder if source == "public40_extra"
            else self.stage39_challenger_decoder
        )
        second_reg, second_cls = self._run_policy_rollout(
            second_policy, extra_initial, ego_query, agents_query,
            bev_feature, bev_spatial_shape, status_encoding, global_img,
        )
        final_poses_reg = torch.cat((public_reg, second_reg), dim=1)
        final_poses_cls = torch.cat((public_cls, second_cls), dim=1)
        num_modes = final_poses_cls.shape[1]
        mode_idx = final_poses_cls.argmax(dim=-1)
        batch_idx = torch.arange(bs, device=device)
        best_reg = final_poses_reg[batch_idx, mode_idx]

        rewards = reward_valid_mask = reward_component_scores = None
        if tokens_list is not None:
            # Keep the public20 branch bitwise invariant when it is embedded
            # in the 40-candidate bank. PDMScorer normalizes progress by the
            # maximum raw progress over the complete proposal set; adding the
            # extra20 therefore changes public20 rewards even when its
            # trajectories are identical. Score both equally sized banks
            # independently, then concatenate their per-candidate results.
            def score_bank(bank_reg):
                if self._lazy_metric_cache is not None:
                    return self._compute_rewards_from_lazy_cache(
                        bank_reg, tokens_list, bank_reg.shape[1]
                    )
                if self.metric_cache_loader is not None:
                    return self._compute_rewards_from_disk(
                        bank_reg, tokens_list, bank_reg.shape[1]
                    )
                return None

            public_result = score_bank(public_reg)
            extra_result = score_bank(second_reg)
            if public_result is not None and extra_result is not None:
                rewards = torch.cat(
                    (public_result["raw_rewards"], extra_result["raw_rewards"]),
                    dim=1,
                )
                reward_valid_mask = torch.cat(
                    (public_result["valid_mask"], extra_result["valid_mask"]),
                    dim=1,
                )
                reward_component_scores = torch.cat(
                    (
                        public_result["component_scores"],
                        extra_result["component_scores"],
                    ),
                    dim=1,
                )

        origin_mask = torch.zeros((bs, num_modes), dtype=torch.bool, device=device)
        origin_mask[:, public_reg.shape[1]:] = True
        return {
            "trajectory": best_reg,
            "final_poses_reg": final_poses_reg,
            "final_poses_cls": final_poses_cls,
            "final_ref_poses_cls": final_poses_cls,
            "final_old_poses_cls": final_poses_cls,
            "inference_selector_logits": final_poses_cls,
            "inference_selector_source": self._inference_selector_source,
            "mode_idx": mode_idx,
            "current_mode_idx": mode_idx,
            "reference_mode_idx": mode_idx,
            "rewards": rewards,
            "reward_valid_mask": reward_valid_mask,
            "reward_component_scores": reward_component_scores,
            "paired_base_poses_reg": None,
            "paired_base_rewards": None,
            "paired_base_valid": None,
            "paired_base_components": None,
            "num_modes": num_modes,
            "grpo_training_rollout": False,
            "stage39_origin_mask": origin_mask,
            "stage39_candidate_source": source,
        }

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img,tokens_list=None) -> Dict[str, torch.Tensor]:
        if (
            self._generation_policy_algorithm == "diffgrpo_non_destructive_challenger"
            and str(getattr(self._config, "stage39_candidate_source", "public20"))
            != "public20"
        ):
            return self._forward_test_stage39_candidates(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, global_img, tokens_list,
            )
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(
            self._scheduler_num_inference_steps, device
        )
        roll_timesteps = torch.tensor(
            self._roll_timesteps, device=device, dtype=torch.long
        )


        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        img = self.norm_odo(plan_anchor)
        if self._generation_trust_collect_calibration and tokens_list is None:
            raise RuntimeError("Trust calibration collection requires scene tokens")
        if self._generation_trust_collect_calibration:
            noise_namespace = self._generation_trust_calibration_seed
        elif self._evaluation_noise_namespace >= 0:
            noise_namespace = self._evaluation_noise_namespace
        else:
            noise_namespace = None
        noise = self._sample_evaluation_noise(img, tokens_list, noise_namespace)
        trunc_timesteps = torch.full(
            (bs,), self._truncation_timestep, device=device, dtype=torch.long
        )
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        initial_sample = img.detach().clone()
        trust_active = (
            self._generation_trust_projection_mode == "reference_mean_ball"
            or self._generation_trust_collect_calibration
        )
        trust_diagnostics = {}
        paired_reference_cls = None
        if trust_active:
            (
                poses_reg,
                poses_cls,
                paired_reference_cls,
                trust_diagnostics,
            ) = self._run_policy_rollout_with_reference(
                initial_sample, ego_query, agents_query, bev_feature,
                bev_spatial_shape, status_encoding, global_img,
                apply_projection=(
                    self._generation_trust_projection_mode == "reference_mean_ball"
                ),
            )
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in (() if trust_active else roll_timesteps[:]):
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
            
        # 获取最终预测
        final_poses_reg = poses_reg  # 已经是最后一个了
        final_poses_cls = poses_cls
        num_modes = final_poses_cls.shape[1]
        
        final_ref_poses_cls = (
            paired_reference_cls
            if trust_active
            else final_poses_cls
        )
        final_ref_poses_reg = None
        if (
            not trust_active
            and self.ref_policy is not None
            and self._inference_selector_source
            in {"reference", "value_top2", "paired_tail_risk"}
        ):
            with torch.no_grad():
                final_ref_poses_reg, final_ref_poses_cls = self._run_policy_rollout(
                    self.ref_policy, initial_sample, ego_query, agents_query,
                    bev_feature, bev_spatial_shape, status_encoding, global_img,
                )

        if (
            self._inference_selector_source == "reference"
            and self.ref_policy is None
        ):
            raise RuntimeError(
                "reference inference selector requires a frozen reference policy"
            )
        value_diagnostics = {}
        stage23_diagnostics = {}
        stage24_diagnostics = {}
        paired_diagnostics = {}
        paired_selected_trajectory = None
        if self._inference_selector_source == "value_top2":
            margin = float(
                getattr(self._config, "value_selector_calibration_margin", -1.0)
            )
            if int(self._value_selector_training_updates.item()) <= 0:
                raise RuntimeError(
                    "value_top2 requires a trained value-selector checkpoint"
                )
            value_outputs = self.value_selector(
                final_poses_reg,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                status_encoding,
            )
            mode_idx, value_diagnostics = select_conservative_top2(
                final_ref_poses_cls,
                value_outputs["component_mean"],
                value_outputs["component_std"],
                value_outputs["score_mean"],
                margin=margin,
                safety_threshold=float(
                    getattr(self._config, "value_selector_safety_threshold", 0.9)
                ),
                confidence_z=float(
                    getattr(self._config, "value_selector_confidence_z", 1.64)
                ),
            )
            inference_selector_logits = torch.full_like(
                final_ref_poses_cls, torch.finfo(final_ref_poses_cls.dtype).min
            )
            inference_selector_logits.scatter_(
                1, mode_idx.unsqueeze(-1), torch.zeros_like(mode_idx, dtype=final_ref_poses_cls.dtype).unsqueeze(-1)
            )
            value_diagnostics.update(
                {
                    "component_predictions": value_outputs["component_predictions"],
                    "score_predictions": value_outputs["score_predictions"],
                    "component_mean": value_outputs["component_mean"],
                    "component_std": value_outputs["component_std"],
                    "score_mean": value_outputs["score_mean"],
                    "score_std": value_outputs["score_std"],
                }
            )
        elif self._inference_selector_source == "trajectory_oof":
            if int(self._stage23_selector_training_updates.item()) <= 0:
                raise RuntimeError(
                    "trajectory_oof requires a trained Stage23 selector checkpoint"
                )
            if num_modes != 20:
                raise RuntimeError("trajectory_oof requires all 20 generator modes")
            stage23_outputs = self.stage23_selector(
                final_poses_reg, final_poses_cls.detach(), bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            mode_idx, stage23_diagnostics = select_stage23_trajectory(
                final_poses_cls.detach(),
                stage23_outputs["component_predictions"],
                stage23_outputs["score_predictions"],
                residual_margin=float(
                    getattr(self._config, "stage23_selector_residual_margin", -1.0)
                ),
                safety_threshold=float(
                    getattr(self._config, "stage23_selector_safety_threshold", 0.95)
                ),
                confidence_z=float(
                    getattr(self._config, "stage23_selector_confidence_z", 1.96)
                ),
                safety_relative_tolerance=float(
                    getattr(self._config, "stage23_selector_safety_tolerance", 0.0)
                ),
            )
            inference_selector_logits = torch.full_like(
                final_poses_cls, torch.finfo(final_poses_cls.dtype).min
            )
            inference_selector_logits.scatter_(
                1, mode_idx.unsqueeze(-1),
                torch.zeros_like(mode_idx, dtype=final_poses_cls.dtype).unsqueeze(-1),
            )
            stage23_diagnostics.update(stage23_outputs)
        elif self._inference_selector_source == "trajectory_relative_harm_v3":
            if (
                int(self._stage24_selector_training_updates.item()) <= 0
                or int(self._stage25_selector_training_updates.item()) <= 0
            ):
                raise RuntimeError(
                    "trajectory_relative_harm_v3 requires trained Stage24/25 state"
                )
            if num_modes != 20:
                raise RuntimeError(
                    "trajectory_relative_harm_v3 requires all 20 modes"
                )
            stage24_outputs = self.stage24_selector(
                final_poses_reg, final_poses_cls.detach(), bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            stage25_outputs = self.stage25_selector(
                stage24_outputs["embedding_predictions"]
            )
            collect_calibration = bool(
                getattr(self._config, "stage25_collect_calibration", False)
            )
            if collect_calibration:
                mode_idx = stage24_outputs["fallback_mode"]
                stage24_diagnostics = {
                    "calibration_collection": torch.ones(
                        bs, dtype=torch.bool, device=device
                    ),
                    "fallback_mode": mode_idx,
                    "switch": torch.zeros(bs, dtype=torch.bool, device=device),
                }
            else:
                if not bool(self._stage24_calibration_loaded.item()):
                    raise RuntimeError(
                        "Stage25 deployment requires loaded calibration/OOD state"
                    )
                mode_idx, stage24_diagnostics = select_stage25_trajectory(
                    final_poses_cls.detach(),
                    stage25_outputs["harm_probabilities"],
                    stage25_outputs["catastrophe_probabilities"],
                    stage24_outputs["delta_predictions"],
                    stage24_outputs["embedding_mean"],
                    residual_margin=float(getattr(
                        self._config, "stage24_selector_residual_margin", -1.0
                    )),
                    risk_threshold=float(getattr(
                        self._config, "stage25_selector_risk_threshold", -1.0
                    )),
                    ood_mean=self._stage24_ood_mean,
                    ood_variance=self._stage24_ood_variance,
                    ood_threshold=float(getattr(
                        self._config, "stage24_selector_ood_threshold", -1.0
                    )),
                    confidence_z=float(getattr(
                        self._config, "stage24_selector_confidence_z", 1.96
                    )),
                )
            inference_selector_logits = torch.full_like(
                final_poses_cls, torch.finfo(final_poses_cls.dtype).min
            )
            inference_selector_logits.scatter_(
                1, mode_idx.unsqueeze(-1),
                torch.zeros_like(
                    mode_idx, dtype=final_poses_cls.dtype
                ).unsqueeze(-1),
            )
            stage24_diagnostics.update(stage24_outputs)
            stage24_diagnostics.update(stage25_outputs)
        elif self._inference_selector_source == "joint_feasible_improvement_v1":
            if (
                int(self._stage24_selector_training_updates.item()) <= 0
                or int(self._stage37_jfi_training_updates.item()) <= 0
            ):
                raise RuntimeError(
                    "joint_feasible_improvement_v1 requires trained Stage24/JFI state"
                )
            if num_modes != 20:
                raise RuntimeError("Stage37 JFI requires all 20 modes")
            stage24_outputs = self.stage24_selector(
                final_poses_reg, final_poses_cls.detach(), bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            jfi_outputs = self.stage37_jfi_selector(
                stage24_outputs["embedding_predictions"]
            )
            if bool(getattr(
                self._config, "stage37_jfi_collect_calibration", False
            )):
                mode_idx = stage24_outputs["fallback_mode"]
                stage24_diagnostics = {
                    "calibration_collection": torch.ones(
                        bs, dtype=torch.bool, device=device
                    ),
                    "fallback_mode": mode_idx,
                    "switch": torch.zeros(
                        bs, dtype=torch.bool, device=device
                    ),
                }
            else:
                if not bool(self._stage24_calibration_loaded.item()):
                    raise RuntimeError("Stage37 JFI requires calibrated OOD state")
                mode_idx, stage24_diagnostics = select_stage37_jfi_trajectory(
                    reference_logits=final_poses_cls.detach(),
                    joint_probabilities=jfi_outputs["joint_probabilities"],
                    delta_quantiles=jfi_outputs["delta_quantiles"],
                    embedding_mean=stage24_outputs["embedding_mean"],
                    joint_threshold=float(getattr(
                        self._config, "stage37_jfi_joint_threshold", -1.0
                    )),
                    q10_floor=float(getattr(
                        self._config, "stage37_jfi_q10_floor", -1.0
                    )),
                    ood_mean=self._stage24_ood_mean,
                    ood_variance=self._stage24_ood_variance,
                    ood_threshold=float(getattr(
                        self._config, "stage24_selector_ood_threshold", -1.0
                    )),
                    confidence_z=float(getattr(
                        self._config, "stage24_selector_confidence_z", 1.96
                    )),
                    max_candidates=int(getattr(
                        self._config, "stage37_jfi_max_candidates", 4
                    )),
                )
            inference_selector_logits = torch.full_like(
                final_poses_cls, torch.finfo(final_poses_cls.dtype).min
            )
            inference_selector_logits.scatter_(
                1, mode_idx.unsqueeze(-1),
                torch.zeros_like(
                    mode_idx, dtype=final_poses_cls.dtype
                ).unsqueeze(-1),
            )
            stage24_diagnostics.update(stage24_outputs)
            stage24_diagnostics.update(jfi_outputs)
        elif self._inference_selector_source == "trajectory_safety_value_v2":
            if int(self._stage24_selector_training_updates.item()) <= 0:
                raise RuntimeError(
                    "trajectory_safety_value_v2 requires a trained Stage24 selector"
                )
            if num_modes != 20:
                raise RuntimeError(
                    "trajectory_safety_value_v2 requires all 20 generator modes"
                )
            stage24_outputs = self.stage24_selector(
                final_poses_reg, final_poses_cls.detach(), bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            collect_calibration = bool(
                getattr(self._config, "stage24_collect_calibration", False)
            )
            if collect_calibration:
                mode_idx = stage24_outputs["fallback_mode"]
                stage24_diagnostics = {
                    "calibration_collection": torch.ones(
                        bs, dtype=torch.bool, device=device
                    ),
                    "fallback_mode": mode_idx,
                    "switch": torch.zeros(bs, dtype=torch.bool, device=device),
                }
            else:
                if not bool(self._stage24_calibration_loaded.item()):
                    raise RuntimeError(
                        "Stage24 deployment requires loaded calibration/OOD state"
                    )
                mode_idx, stage24_diagnostics = select_stage24_trajectory(
                    final_poses_cls.detach(),
                    stage24_outputs["safety_probabilities"],
                    stage24_outputs["any_unsafe_probabilities"],
                    stage24_outputs["component_delta_predictions"],
                    stage24_outputs["delta_predictions"],
                    stage24_outputs["embedding_mean"],
                    residual_margin=float(getattr(
                        self._config, "stage24_selector_residual_margin", -1.0
                    )),
                    risk_threshold=float(getattr(
                        self._config, "stage24_selector_risk_threshold", -1.0
                    )),
                    ood_mean=self._stage24_ood_mean,
                    ood_variance=self._stage24_ood_variance,
                    ood_threshold=float(getattr(
                        self._config, "stage24_selector_ood_threshold", -1.0
                    )),
                    confidence_z=float(getattr(
                        self._config, "stage24_selector_confidence_z", 1.96
                    )),
                    safety_tolerance=float(getattr(
                        self._config, "stage24_selector_safety_tolerance", 0.001
                    )),
                )
            inference_selector_logits = torch.full_like(
                final_poses_cls, torch.finfo(final_poses_cls.dtype).min
            )
            inference_selector_logits.scatter_(
                1, mode_idx.unsqueeze(-1),
                torch.zeros_like(
                    mode_idx, dtype=final_poses_cls.dtype
                ).unsqueeze(-1),
            )
            stage24_diagnostics.update(stage24_outputs)
        elif self._inference_selector_source == "paired_tail_risk":
            if final_ref_poses_reg is None:
                raise RuntimeError(
                    "paired_tail_risk requires frozen base trajectories"
                )
            if int(self._paired_risk_training_updates.item()) <= 0:
                raise RuntimeError(
                    "paired_tail_risk requires a trained adapter checkpoint"
                )
            threshold = float(
                getattr(self._config, "paired_risk_threshold", -1.0)
            )
            risk_outputs = self.paired_risk_head(
                final_poses_reg, final_ref_poses_reg, bev_feature,
                bev_spatial_shape, agents_query, ego_query, status_encoding,
            )
            paired_selected_trajectory, paired_diagnostics = (
                select_same_mode_with_base_fallback(
                    final_poses_reg, final_ref_poses_reg,
                    final_ref_poses_cls, risk_outputs["fallback_score"],
                    threshold,
                )
            )
            mode_idx = paired_diagnostics["mode"]
            inference_selector_logits = final_ref_poses_cls.detach()
            paired_diagnostics.update(risk_outputs)
        else:
            mode_idx, inference_selector_logits = select_inference_mode(
                final_poses_cls,
                final_ref_poses_cls,
                self._inference_selector_source,
            )
        batch_idx = torch.arange(bs, device=device)
        best_reg = final_poses_reg[batch_idx, mode_idx]
        if paired_selected_trajectory is not None:
            best_reg = paired_selected_trajectory
        current_mode_idx = final_poses_cls.argmax(dim=-1)
        reference_mode_idx = final_ref_poses_cls.argmax(dim=-1)

        rewards = None
        reward_valid_mask = None
        reward_component_scores = None
        paired_base_rewards = None
        paired_base_valid = None
        paired_base_components = None
        if (
            tokens_list is not None
            and self._inference_selector_source
            not in {
                "trajectory_oof", "trajectory_safety_value_v2",
                "trajectory_relative_harm_v3", "joint_feasible_improvement_v1",
            }
        ):
            reward_trajectories = final_poses_reg
            reward_modes = num_modes
            if self._inference_selector_source == "paired_tail_risk":
                reward_trajectories = torch.cat(
                    (final_poses_reg, final_ref_poses_reg), dim=1
                )
                reward_modes = 2 * num_modes
            if self._lazy_metric_cache is not None:
                reward_result = self._compute_rewards_from_lazy_cache(
                    reward_trajectories, tokens_list, reward_modes,
                )
            elif self.metric_cache_loader is not None:
                reward_result = self._compute_rewards_from_disk(
                    reward_trajectories, tokens_list, reward_modes,
                )
            else:
                reward_result = None
            if reward_result is not None:
                # Evaluation and checkpoint selection always use aggregate
                # PDMS, even when training used the component tie-break.
                rewards = reward_result["raw_rewards"][:, :num_modes]
                reward_valid_mask = reward_result["valid_mask"][:, :num_modes]
                reward_component_scores = reward_result["component_scores"][:, :num_modes]
                if self._inference_selector_source == "paired_tail_risk":
                    paired_base_rewards = reward_result["raw_rewards"][:, num_modes:]
                    paired_base_valid = reward_result["valid_mask"][:, num_modes:]
                    paired_base_components = reward_result["component_scores"][:, num_modes:]

        output_dict = {
            "trajectory": best_reg,
            "final_poses_reg": final_poses_reg,
            "final_poses_cls": final_poses_cls,
            "final_ref_poses_cls": final_ref_poses_cls,
            "final_old_poses_cls": final_poses_cls,
            "inference_selector_logits": inference_selector_logits,
            "inference_selector_source": self._inference_selector_source,
            "mode_idx": mode_idx,
            "current_mode_idx": current_mode_idx,
            "reference_mode_idx": reference_mode_idx,
            "rewards": rewards,
            "reward_valid_mask": reward_valid_mask,
            "reward_component_scores": reward_component_scores,
            "paired_base_poses_reg": final_ref_poses_reg,
            "paired_base_rewards": paired_base_rewards,
            "paired_base_valid": paired_base_valid,
            "paired_base_components": paired_base_components,
            "num_modes": num_modes,
            "grpo_training_rollout": False,
            **{f"value_selector_{key}": value for key, value in value_diagnostics.items()},
            **{f"stage23_selector_{key}": value for key, value in stage23_diagnostics.items()},
            **{f"stage24_selector_{key}": value for key, value in stage24_diagnostics.items()},
            **{f"paired_risk_{key}": value for key, value in paired_diagnostics.items()},
            **trust_diagnostics,
        }
        #print(f"[TEST] 返回 - rewards: {output_dict['rewards']}, 其他keys: {list(output_dict.keys())}")
        return output_dict

    def _compute_rewards_from_lazy_cache(
        self, trajectories: torch.Tensor, tokens_list, num_modes: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards using lazily loaded metric caches (first hit reads disk, then cached)."""
        cache_dict = {}
        for token in set(tokens_list):
            cache_dict[token] = self._get_metric_cache_lazy(token)
        return self._reward_fn_with_cache(trajectories, tokens_list, num_modes, cache_dict)

    def _compute_rewards_from_disk(
        self, trajectories: torch.Tensor, tokens_list, num_modes: int,
    ) -> Dict[str, torch.Tensor]:
        """Original fallback: load metric caches from disk per step (slowest)."""
        cache_dict = {}
        for token in set(tokens_list):
            try:
                path = self.metric_cache_loader.metric_cache_paths.get(token)
                if path is None:
                    cache_dict[token] = None
                    continue
                with lzma.open(path, "rb") as f:
                    cache_dict[token] = pickle.load(f)
            except Exception:
                cache_dict[token] = None
        return self._reward_fn_with_cache(trajectories, tokens_list, num_modes, cache_dict)

    def _reward_fn_with_cache(
        self,
        trajectories: torch.Tensor,
        tokens_list,
        num_modes: int,
        cache_dict: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Batched PDM reward: simulate all modes per token in one call instead of one-by-one."""
        batch_size = trajectories.shape[0]
        pred_np = trajectories.reshape(-1, 8, 3).detach().cpu().numpy()
        rewards = np.full(batch_size * num_modes, np.nan, dtype=np.float32)
        components = np.full(
            (batch_size * num_modes, 6), np.nan, dtype=np.float32
        )
        valid_mask = np.zeros(batch_size * num_modes, dtype=np.bool_)
        future_sampling = self.simulator.proposal_sampling

        for batch_idx, token in enumerate(tokens_list):
            metric_cache = cache_dict.get(token)
            if metric_cache is None:
                continue

            try:
                initial_ego_state = metric_cache.ego_state

                # PDM reference (computed once per token, not once per mode)
                pdm_states = get_trajectory_as_array(
                    metric_cache.trajectory, future_sampling, initial_ego_state.time_point,
                )

                mode_start = batch_idx * num_modes
                pred_states_list = []
                valid_mode_indices = []

                for mode_idx in range(num_modes):
                    try:
                        traj_np = pred_np[mode_start + mode_idx]
                        model_traj = Trajectory(traj_np)
                        pred_traj = transform_trajectory(model_traj, initial_ego_state)
                        pred_states = get_trajectory_as_array(
                            pred_traj, future_sampling, initial_ego_state.time_point,
                        )
                        pred_states_list.append(pred_states)
                        valid_mode_indices.append(mode_idx)
                    except Exception:
                        pass

                if not valid_mode_indices:
                    continue

                # Stack: [1 + valid_count, timesteps, state_dim]
                all_states = np.concatenate(
                    [pdm_states[None, ...]] + [s[None, ...] for s in pred_states_list],
                    axis=0,
                )

                # Single batched simulate + score call for all modes of this token
                simulated_states = self.simulator.simulate_proposals(all_states, initial_ego_state)
                scores = self.scorer.score_proposals(
                    simulated_states,
                    metric_cache.observation,
                    metric_cache.centerline,
                    metric_cache.route_lane_ids,
                    metric_cache.drivable_area_map,
                )
                multi_metrics = self.scorer._multi_metrics[:, 1:].copy()
                weighted_metrics = self.scorer._weighted_metrics[:, 1:].copy()

                # scores[0] = pdm reference; scores[1:] = predicted modes
                for j, mode_idx in enumerate(valid_mode_indices):
                    score = float(scores[j + 1])
                    component_values = np.asarray(
                        [
                            multi_metrics[MultiMetricIndex.NO_COLLISION, j],
                            multi_metrics[MultiMetricIndex.DRIVABLE_AREA, j],
                            weighted_metrics[WeightedMetricIndex.PROGRESS, j],
                            weighted_metrics[WeightedMetricIndex.TTC, j],
                            weighted_metrics[WeightedMetricIndex.COMFORTABLE, j],
                            weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, j],
                        ],
                        dtype=np.float32,
                    )
                    if np.isfinite(score) and np.isfinite(component_values).all():
                        rewards[mode_start + mode_idx] = score
                        components[mode_start + mode_idx] = component_values
                        valid_mask[mode_start + mode_idx] = True

            except Exception:
                pass

        raw_rewards = torch.tensor(
            rewards, device=trajectories.device, dtype=trajectories.dtype,
        ).detach().view(batch_size, num_modes)
        component_tensor = torch.tensor(
            components, device=trajectories.device, dtype=trajectories.dtype,
        ).detach().view(batch_size, num_modes, 6)
        valid_mask = torch.tensor(
            valid_mask, device=trajectories.device, dtype=torch.bool,
        ).detach().view(batch_size, num_modes)
        weighted_metric_weights = torch.as_tensor(
            self.scorer._config.weighted_metrics_array,
            device=trajectories.device,
            dtype=trajectories.dtype,
        )
        shaped_rewards, secondary_score, tie_epsilon = compute_pdm_tiebreak_rewards(
            aggregate_rewards=raw_rewards,
            component_scores=component_tensor,
            valid_mask=valid_mask,
            weighted_metric_weights=weighted_metric_weights,
            max_epsilon=self._pdm_tiebreak_max_epsilon,
        )
        dense_rewards, _, _ = compute_pdm_dense_rewards(
            aggregate_rewards=raw_rewards,
            component_scores=component_tensor,
            valid_mask=valid_mask,
            weighted_metric_weights=weighted_metric_weights,
            dense_weight=self._pdm_dense_weight,
        )
        if self._grpo_reward_mode == "pdm_tiebreak":
            training_rewards = shaped_rewards
        elif self._grpo_reward_mode == "pdm_dense":
            training_rewards = dense_rewards
        else:
            training_rewards = raw_rewards
        return {
            "training_rewards": training_rewards,
            "raw_rewards": raw_rewards,
            "valid_mask": valid_mask,
            "component_scores": component_tensor,
            "secondary_score": secondary_score.detach(),
            "tiebreak_epsilon": tie_epsilon.detach(),
        }
