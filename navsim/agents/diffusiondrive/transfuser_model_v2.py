from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import copy
import hashlib
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

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score, transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
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
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
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
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
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
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.ref_policy = None
        self.old_policy = None
        self._old_policy_sync_steps = int(getattr(config, "grpo_old_policy_sync_steps", 32))
        self._policy_forward_steps = 0

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

    @torch.no_grad()
    def maybe_sync_old_policy(self):
        if self.old_policy is None:
            return
        if self._policy_forward_steps % self._old_policy_sync_steps == 0:
            self.old_policy.load_state_dict(self.diff_decoder.state_dict())
            self.old_policy.eval()
        self._policy_forward_steps += 1
    
    
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
        self.diffusion_scheduler.set_timesteps(1000, device)
        roll_timesteps = torch.tensor([10, 0], device=device, dtype=torch.long)
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

    def forward_train_grpo(
        self, ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, targets=None, global_img=None, tokens_list=None,
    ) -> Dict[str, torch.Tensor]:
        """Two-step mode-selection rollout with current, old and reference policies."""
        if self.ref_policy is None or self.old_policy is None:
            raise RuntimeError("GRPO requires both reference and old policies")

        self.maybe_sync_old_policy()
        self.diff_decoder.eval()
        self.ref_policy.eval()
        self.old_policy.eval()

        bs = ego_query.shape[0]
        device = ego_query.device
        plan_anchor = self.plan_anchor.unsqueeze(0).expand(bs, -1, -1, -1)
        normalized_anchor = self.norm_odo(plan_anchor)
        trunc_timesteps = torch.full((bs,), 8, device=device, dtype=torch.long)
        initial_sample = self.diffusion_scheduler.add_noise(
            original_samples=normalized_anchor,
            noise=torch.randn_like(normalized_anchor),
            timesteps=trunc_timesteps,
        )

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
        reward_valid_mask = None
        num_modes = final_poses_cls.shape[-1]
        if tokens_list is not None:
            if self._lazy_metric_cache is not None:
                reward_result = self._compute_rewards_from_lazy_cache(
                    final_poses_reg, tokens_list, num_modes,
                )
            elif self.metric_cache_loader is not None:
                reward_result = self._compute_rewards_from_disk(
                    final_poses_reg, tokens_list, num_modes,
                )
            else:
                reward_result = None
            if reward_result is not None:
                if isinstance(reward_result, tuple):
                    rewards, reward_valid_mask = reward_result
                else:
                    rewards = reward_result
                    reward_valid_mask = torch.isfinite(rewards)
        if rewards is None or reward_valid_mask is None:
            raise RuntimeError(
                "GRPO training requires PDM rewards and scene tokens for every batch"
            )
        if not reward_valid_mask.any():
            raise RuntimeError(
                f"No valid PDM reward in batch; tokens={list(tokens_list or [])}"
            )

        mode_idx = final_poses_cls.argmax(dim=-1)
        best_reg = final_poses_reg[torch.arange(bs, device=device), mode_idx]
        return {
            "trajectory": best_reg,
            "final_poses_cls": final_poses_cls,
            "final_old_poses_cls": final_old_poses_cls,
            "final_ref_poses_cls": final_ref_poses_cls,
            "rewards": rewards,
            "reward_valid_mask": reward_valid_mask,
            "num_modes": num_modes,
            "mode_idx": mode_idx,
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

    def _sample_evaluation_noise(self, template: torch.Tensor, tokens_list) -> torch.Tensor:
        """Generate order-independent noise from scenario tokens during evaluation."""
        if tokens_list is None or len(tokens_list) != template.shape[0]:
            return torch.randn_like(template)

        samples = []
        for token in tokens_list:
            digest = hashlib.sha256(str(token).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="little") % (2**63 - 1)
            generator = torch.Generator(device=template.device)
            generator.manual_seed(seed)
            samples.append(
                torch.randn(template.shape[1:], device=template.device, dtype=template.dtype, generator=generator)
            )
        return torch.stack(samples, dim=0)

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img,tokens_list=None) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        #step_ratio = 64 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        img = self.norm_odo(plan_anchor)
        noise = self._sample_evaluation_noise(img, tokens_list)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        initial_sample = img.detach().clone()
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
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
        
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)

        final_ref_poses_cls = final_poses_cls
        if self.ref_policy is not None:
            with torch.no_grad():
                _, final_ref_poses_cls = self._run_policy_rollout(
                    self.ref_policy, initial_sample, ego_query, agents_query,
                    bev_feature, bev_spatial_shape, status_encoding, global_img,
                )

        rewards = None
        reward_valid_mask = None
        if tokens_list is not None:
            if self._lazy_metric_cache is not None:
                reward_result = self._compute_rewards_from_lazy_cache(
                    final_poses_reg, tokens_list, num_modes,
                )
            elif self.metric_cache_loader is not None:
                reward_result = self._compute_rewards_from_disk(
                    final_poses_reg, tokens_list, num_modes,
                )
            else:
                reward_result = None
            if reward_result is not None:
                if isinstance(reward_result, tuple):
                    rewards, reward_valid_mask = reward_result
                else:
                    rewards = reward_result
                    reward_valid_mask = torch.isfinite(rewards)

        output_dict = {
        "trajectory": best_reg,                    # 主输出（必需）
        "final_poses_cls": final_poses_cls,        # GRPO损失必需
        "final_ref_poses_cls": final_ref_poses_cls,
        "final_old_poses_cls": final_poses_cls,
        "rewards": rewards,
        "reward_valid_mask": reward_valid_mask,
        "num_modes": num_modes                     # 
        }
        #print(f"[TEST] 返回 - rewards: {output_dict['rewards']}, 其他keys: {list(output_dict.keys())}")
        return output_dict

    def _compute_rewards_from_lazy_cache(
        self, trajectories: torch.Tensor, tokens_list, num_modes: int,
    ) -> torch.Tensor:
        """Compute rewards using lazily loaded metric caches (first hit reads disk, then cached)."""
        cache_dict = {}
        for token in set(tokens_list):
            cache_dict[token] = self._get_metric_cache_lazy(token)
        return self._reward_fn_with_cache(trajectories, tokens_list, num_modes, cache_dict)

    def _compute_rewards_from_disk(
        self, trajectories: torch.Tensor, tokens_list, num_modes: int,
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        """Batched PDM reward: simulate all modes per token in one call instead of one-by-one."""
        batch_size = trajectories.shape[0]
        pred_np = trajectories.reshape(-1, 8, 3).detach().cpu().numpy()
        rewards = np.full(batch_size * num_modes, np.nan, dtype=np.float32)
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

                # scores[0] = pdm reference; scores[1:] = predicted modes
                for j, mode_idx in enumerate(valid_mode_indices):
                    score = float(scores[j + 1])
                    if np.isfinite(score):
                        rewards[mode_start + mode_idx] = score
                        valid_mask[mode_start + mode_idx] = True

            except Exception:
                pass

        rewards_tensor = torch.tensor(
            rewards, device=trajectories.device, dtype=trajectories.dtype,
        ).detach()
        valid_mask_tensor = torch.tensor(
            valid_mask, device=trajectories.device, dtype=torch.bool,
        )
        return (
            rewards_tensor.view(batch_size, num_modes),
            valid_mask_tensor.view(batch_size, num_modes),
        )
