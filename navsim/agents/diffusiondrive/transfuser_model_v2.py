from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import copy
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
from diffusers.utils.torch_utils import randn_tensor
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


class DDIMScheduler_with_logprob(DDIMScheduler):
    """DDIMScheduler extended to return log_prob of prev_sample under the Gaussian transition."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, None]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** (0.5)).clamp_(min=1e-10)

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).clamp_(min=0) ** (0.5) * pred_epsilon
        prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if prev_sample is None:
            if eta > 0:
                std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            else:
                std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)

            variance_noise_horizon = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype,
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype,
            ) * std_dev_t_mul + 1.0

            variance_noise_mul = torch.cat((variance_noise_horizon, variance_noise_vert), dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1, 1, model_output.shape[2], 1)

            variance_noise_x = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype,
            )
            variance_noise_y = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype,
            )
            variance_noise_add = torch.cat((variance_noise_x, variance_noise_y), dim=-1)
            variance_noise_add = variance_noise_add.repeat(1, 1, model_output.shape[2], 1)

            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add

        std_dev_t_mul = torch.clip(std_dev_t, min=0.1)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul ** 2))
            - torch.log(std_dev_t_mul)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.sum(dim=(-2, -1))

        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)


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

        if self.training:
            with torch.no_grad():
                old_pred = self._trajectory_head(
                    trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
                    status_encoding[:, None], targets=targets, global_img=None,
                    tokens_list=tokens_list, old_pred=None,
                )
            trajectory = self._trajectory_head(
                trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
                status_encoding[:, None], targets=targets, global_img=None,
                tokens_list=tokens_list, old_pred=old_pred,
            )
            if 'reward' not in trajectory:
                trajectory['reward'] = old_pred.get('reward')
        else:
            trajectory = self._trajectory_head(
                trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
                status_encoding[:, None], targets=targets, global_img=None,
                tokens_list=tokens_list,
            )
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

        self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
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
        #self。ref decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.ref_policy = None

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

        # Reward skip: only compute PDM rewards every N steps, reuse cached rewards in between.
        self._reward_compute_interval: int = int(getattr(config, "reward_compute_interval", 1))
        self._reward_step_counter: int = 0
        self._cached_rewards: Optional[torch.Tensor] = None
        if self._reward_compute_interval > 1:
            print(f"Reward skip: compute PDM rewards every {self._reward_compute_interval} steps, reuse in between.")

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
    
    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None, tokens_list=None, old_pred=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            if old_pred is not None:
                return self.get_rlloss(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets, global_img, old_pred)
            else:
                return self.forward_train_rl(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets, global_img, tokens_list)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, global_img, tokens_list)


    def forward_train_rl(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        """Pass 1 (no_grad): run full denoising chain, collect chain states, compute rewards & advantages."""
        step_num = 10
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        diffusion_output = self.norm_odo(plan_anchor)

        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusionrl_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps,
        )

        all_log_probs = []
        all_diffusion_output = [diffusion_output]
        ego_fut_mode = diffusion_output.shape[1]

        for i, k in enumerate(roll_timesteps):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k.expand(bs)
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img,
            )
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)

            prev_sample, log_prob, _ = self.diffusionrl_scheduler.step(
                model_output=x_start, timestep=k, sample=diffusion_output, eta=1.0,
            )
            diffusion_output = prev_sample
            all_log_probs.append(log_prob)
            all_diffusion_output.append(prev_sample)

        all_log_probs = torch.stack(all_log_probs, dim=-1)
        all_diffusion_output = torch.stack(all_diffusion_output, dim=-1)

        final_traj = poses_reg
        num_modes = ego_fut_mode

        rewards = None
        if tokens_list is not None:
            if self._lazy_metric_cache is not None:
                rewards = self._compute_rewards_from_lazy_cache(final_traj, tokens_list, num_modes)
            elif self.metric_cache_loader is not None:
                rewards = self._compute_rewards_from_disk(final_traj, tokens_list, num_modes)

        advantages = None
        reward_mean = None
        if rewards is not None:
            mean_r = rewards.mean(dim=1, keepdim=True)
            std_r = rewards.std(dim=1, keepdim=True).clamp(min=1e-4)
            advantages = ((rewards - mean_r) / std_r).detach()
            advantages = advantages.unsqueeze(-1).repeat(1, 1, step_num)
            discount = torch.tensor(
                [0.8 ** (step_num - i - 1) for i in range(step_num)]
            ).to(device)
            advantages = advantages * discount
            reward_mean = rewards.mean()

        target_traj = targets["trajectory"]
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - self.plan_anchor.unsqueeze(0), dim=-1)
        mode_idx = torch.argmin(dist.mean(dim=-1), dim=-1)
        best_reg = poses_reg[torch.arange(bs), mode_idx]

        return {
            "all_diffusion_output": all_diffusion_output,
            "advantages": advantages,
            "reward": reward_mean,
            "trajectory": best_reg,
        }

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
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
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
        
        output_dict = {
        "trajectory": best_reg,                    # 主输出（必需）
        "final_poses_cls": final_poses_cls,        # GRPO损失必需
        "final_ref_poses_cls": final_poses_cls,    # validation时用相同的
        "rewards": None,                           # validation时不需要奖励
        "num_modes": num_modes                     # 
        }
        return output_dict

    def get_rlloss(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets, global_img, old_pred) -> Dict[str, torch.Tensor]:
        """Pass 2 (with_grad): replay chain with current policy, compute log_prob, GRPO loss + IL loss."""
        old_diffusion_output = old_pred['all_diffusion_output']
        advantages = old_pred['advantages']

        chains = old_diffusion_output[..., :-1]
        chains_prev = old_diffusion_output[..., 1:]

        step_num = chains.shape[-1]
        bs = chains.shape[0]
        device = chains.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        all_log_probs = []
        poses_reg_steps_list = []

        for i, k in enumerate(roll_timesteps):
            diffusion_input = chains[..., i]
            ego_fut_mode = diffusion_input.shape[1]
            x_boxes = torch.clamp(diffusion_input, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k.expand(bs)
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img,
            )
            poses_reg_steps_list.append(poses_reg_list)
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)

            _, log_prob, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_input,
                eta=1.0,
                prev_sample=chains_prev[..., i],
            )
            all_log_probs.append(log_prob)

        all_log_probs = torch.stack(all_log_probs, dim=-1)
        per_token_logps = all_log_probs

        if advantages is not None:
            per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages
            mask_nz = per_token_loss != 0
            RL_loss_b = (per_token_loss * mask_nz).sum(dim=1) / mask_nz.sum(dim=1).clamp_min(1)
            RL_loss_b = RL_loss_b.mean(dim=-1)
        else:
            RL_loss_b = torch.zeros(bs, device=device)

        IL_loss_b = torch.zeros_like(RL_loss_b)
        target_traj = targets['trajectory'].unsqueeze(1).repeat(1, ego_fut_mode, 1, 1)
        for reg_list in poses_reg_steps_list:
            for poses_reg_layer in reg_list:
                traj_l1 = F.l1_loss(poses_reg_layer[..., :2], target_traj[..., :2], reduction='none')
                IL_loss_b = IL_loss_b + traj_l1.mean(dim=(1, 2, 3))
        IL_loss_b = IL_loss_b / (len(poses_reg_steps_list) * len(poses_reg_steps_list[0]))

        has_positive = (
            (advantages > 0).any(dim=2).any(dim=1)
            if advantages is not None
            else torch.zeros(bs, dtype=torch.bool, device=device)
        )
        il_weight = torch.where(
            has_positive,
            torch.tensor(0.1, device=device),
            torch.tensor(1.0, device=device),
        )
        loss_b = RL_loss_b + il_weight * IL_loss_b
        loss = loss_b.mean()

        last_poses_cls = poses_cls_list[-1]
        mode_idx = last_poses_cls.argmax(dim=-1)
        best_reg = poses_reg_list[-1][torch.arange(bs), mode_idx]

        return {
            "loss": loss,
            "rl_loss": RL_loss_b.mean().detach(),
            "il_loss": IL_loss_b.mean().detach(),
            "trajectory": best_reg,
        }

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
        rewards = np.full(batch_size * num_modes, 0.5, dtype=np.float32)
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
                    rewards[mode_start + mode_idx] = scores[j + 1]

            except Exception:
                pass

        rewards_tensor = torch.tensor(
            rewards, device=trajectories.device, dtype=trajectories.dtype,
        ).detach()
        return rewards_tensor.view(batch_size, num_modes)