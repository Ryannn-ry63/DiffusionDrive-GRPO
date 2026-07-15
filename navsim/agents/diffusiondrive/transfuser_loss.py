from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-3,
):
    """Normalize rewards within each scene while excluding invalid modes."""
    if rewards.ndim != 2 or valid_mask.shape != rewards.shape:
        raise ValueError("rewards and valid_mask must both have shape [batch, num_modes]")

    rewards = rewards.float()
    valid_mask = valid_mask.bool() & torch.isfinite(rewards)
    clean_rewards = torch.where(valid_mask, rewards, torch.zeros_like(rewards))
    counts = valid_mask.sum(dim=-1, keepdim=True)
    safe_counts = counts.clamp_min(1)
    means = clean_rewards.sum(dim=-1, keepdim=True) / safe_counts
    centered = torch.where(valid_mask, clean_rewards - means, torch.zeros_like(rewards))
    variances = centered.square().sum(dim=-1, keepdim=True) / safe_counts
    stds = variances.sqrt()
    group_valid = (counts.squeeze(-1) >= 2) & (stds.squeeze(-1) >= eps)

    advantages = centered / stds.clamp_min(eps)
    advantages = torch.where(
        valid_mask & group_valid.unsqueeze(-1), advantages, torch.zeros_like(advantages)
    ).detach()
    return advantages, valid_mask, group_valid, means.squeeze(-1), stds.squeeze(-1)


def compute_grpo_objective(
    current_logits: torch.Tensor,
    old_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    advantage_eps: float = 1e-3,
    temperature: float = 1.0,
):
    """Exact categorical, group-relative clipped policy objective over all modes."""
    if not (current_logits.shape == old_logits.shape == reference_logits.shape == rewards.shape):
        raise ValueError("policy logits and rewards must have identical [batch, num_modes] shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    current_logits = current_logits.float() / temperature
    old_logits = old_logits.detach().float() / temperature
    reference_logits = reference_logits.detach().float() / temperature
    advantages, valid_mask, group_valid, reward_mean, reward_std = (
        compute_group_relative_advantages(rewards, valid_mask, advantage_eps)
    )

    current_log_probs = F.log_softmax(current_logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    old_weights = old_log_probs.exp() * valid_mask.float()
    old_weights = old_weights / old_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    log_ratio = (current_log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
    )
    per_scene_policy_loss = -(old_weights * surrogate).sum(dim=-1)
    policy_loss = (
        per_scene_policy_loss[group_valid].mean()
        if group_valid.any()
        else current_logits.sum() * 0.0
    )

    current_probs = current_log_probs.exp()
    kl_loss = (current_probs * (current_log_probs - reference_log_probs)).sum(dim=-1).mean()
    entropy = -(current_probs * current_log_probs).sum(dim=-1).mean()

    selected_idx = current_logits.argmax(dim=-1)
    clean_rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
    selected_reward_by_scene = clean_rewards.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1)
    oracle_reward_by_scene = clean_rewards.masked_fill(~valid_mask, float("-inf")).max(dim=-1).values
    has_valid = valid_mask.any(dim=-1)
    selected_is_valid = valid_mask.gather(1, selected_idx.unsqueeze(-1)).squeeze(-1)
    metric_valid = has_valid & selected_is_valid
    selected_reward = (
        selected_reward_by_scene[metric_valid].mean()
        if metric_valid.any() else rewards.new_zeros(())
    )
    oracle_reward = (
        oracle_reward_by_scene[metric_valid].mean()
        if metric_valid.any() else rewards.new_zeros(())
    )

    clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
    oracle_idx = clean_rewards.masked_fill(~valid_mask, float("-inf")).argmax(dim=-1)
    oracle_hit_rate = (
        (selected_idx[metric_valid] == oracle_idx[metric_valid]).float().mean()
        if metric_valid.any() else rewards.new_zeros(())
    )
    clip_fraction_per_scene = (old_weights * clipped.float()).sum(dim=-1)
    clip_fraction = (
        clip_fraction_per_scene[group_valid].mean()
        if group_valid.any()
        else rewards.new_zeros(())
    )

    return {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "entropy_for_loss": entropy,
        "entropy": entropy.detach(),
        "reward_mean": reward_mean.mean().detach(),
        "reward_std": reward_std.mean().detach(),
        "selected_reward": selected_reward.detach(),
        "oracle_reward": oracle_reward.detach(),
        "selection_regret": (oracle_reward - selected_reward).detach(),
        "oracle_hit_rate": oracle_hit_rate.detach(),
        "valid_mode_fraction": valid_mask.float().mean().detach(),
        "valid_group_fraction": group_valid.float().mean().detach(),
        "zero_advantage_fraction": (1.0 - group_valid.float().mean()).detach(),
        "ratio_mean": (old_weights * ratio).sum(dim=-1).mean().detach(),
        "clip_fraction": clip_fraction.detach(),
    }


def compute_generation_grpo_objective(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    generation_kl: torch.Tensor,
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    advantage_eps: float = 1e-3,
):
    """Clipped group-relative objective over sampled denoising actions."""
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old generation log-probs must have identical shapes")
    if current_log_probs.ndim != 3 or current_log_probs.shape[:2] != rewards.shape:
        raise ValueError("generation log-probs must have shape [batch, mode, step]")
    if generation_kl.shape != current_log_probs.shape:
        raise ValueError("generation KL must match generation log-prob shape")

    advantages, valid_mask, group_valid, _, _ = compute_group_relative_advantages(
        rewards, valid_mask, advantage_eps
    )
    log_ratio = (current_log_probs - old_log_probs.detach()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    step_advantages = advantages.unsqueeze(-1)
    surrogate = torch.minimum(
        ratio * step_advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * step_advantages,
    )
    optimize_mask = (
        valid_mask
        & group_valid.unsqueeze(-1)
    ).unsqueeze(-1).expand_as(surrogate)
    policy_loss = (
        -surrogate[optimize_mask].mean()
        if optimize_mask.any()
        else current_log_probs.sum() * 0.0
    )
    kl_mask = valid_mask.unsqueeze(-1).expand_as(generation_kl)
    kl_loss = (
        generation_kl[kl_mask].mean()
        if kl_mask.any()
        else generation_kl.sum() * 0.0
    )
    clipped = (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
    return {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "ratio_mean": ratio[kl_mask].mean().detach(),
        "clip_fraction": clipped[kl_mask].float().mean().detach(),
    }


def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """Pure GRPO objective for selection, generation, or their joint policy."""
    current_logits = predictions["final_poses_cls"]
    grpo_weight = getattr(config, "policy_loss_weight", 1.0)
    kl_weight = getattr(config, "kl_loss_weight", 0.01)
    entropy_weight = getattr(config, "selection_entropy_weight", 0.0)

    if "rewards" not in predictions or predictions["rewards"] is None:
        zero = current_logits.sum() * 0.0
        return {
            "loss": zero,
            "grpo_loss": zero.detach(),
            "kl_loss": zero.detach(),
        }

    if predictions.get("final_old_poses_cls") is None:
        raise ValueError("GRPO training requires final_old_poses_cls")
    if predictions.get("final_ref_poses_cls") is None:
        raise ValueError("GRPO training requires final_ref_poses_cls")

    objective = compute_grpo_objective(
        current_logits=current_logits,
        old_logits=predictions["final_old_poses_cls"],
        reference_logits=predictions["final_ref_poses_cls"],
        rewards=predictions["rewards"],
        valid_mask=predictions.get(
            "reward_valid_mask", torch.ones_like(predictions["rewards"], dtype=torch.bool)
        ),
        clip_ratio=getattr(config, "grpo_clip_ratio", 0.2),
        advantage_eps=getattr(config, "grpo_advantage_eps", 1e-3),
        temperature=getattr(config, "selection_temperature", 1.0),
    )

    training_mode = getattr(config, "grpo_training_mode", "classification_shared")
    selection_weight = 0.0 if training_mode == "generation" else grpo_weight
    total_loss = (
        selection_weight * objective["policy_loss"]
        + kl_weight * objective["kl_loss"]
        - entropy_weight * objective["entropy_for_loss"]
    )

    generation_objective = None
    if (
        training_mode in {"generation", "joint"}
        and predictions.get("generation_current_log_probs") is not None
    ):
        required = (
            "generation_current_log_probs",
            "generation_old_log_probs",
            "generation_kl",
        )
        missing = [key for key in required if predictions.get(key) is None]
        if missing:
            raise ValueError(f"Missing generation GRPO tensors: {missing}")
        generation_objective = compute_generation_grpo_objective(
            current_log_probs=predictions["generation_current_log_probs"],
            old_log_probs=predictions["generation_old_log_probs"],
            generation_kl=predictions["generation_kl"],
            rewards=predictions["rewards"],
            valid_mask=predictions.get(
                "reward_valid_mask",
                torch.ones_like(predictions["rewards"], dtype=torch.bool),
            ),
            clip_ratio=getattr(config, "grpo_clip_ratio", 0.2),
            advantage_eps=getattr(config, "grpo_advantage_eps", 1e-3),
        )
        total_loss = (
            total_loss
            + getattr(config, "generation_policy_loss_weight", 1.0)
            * generation_objective["policy_loss"]
            + getattr(config, "generation_kl_loss_weight", 0.1)
            * generation_objective["kl_loss"]
        )

    result = {
        "loss": total_loss,
        "grpo_loss": selection_weight * objective["policy_loss"],
        "kl_loss": objective["kl_loss"],
        **{
            key: value for key, value in objective.items()
            if key not in {"policy_loss", "kl_loss", "entropy_for_loss"}
        },
    }
    if generation_objective is not None:
        result.update(
            {
                "generation_grpo_loss": generation_objective["policy_loss"],
                "generation_kl_loss": generation_objective["kl_loss"],
                "generation_ratio_mean": generation_objective["ratio_mean"],
                "generation_clip_fraction": generation_objective["clip_fraction"],
            }
        )
    return result


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def compute_grpo_loss(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    num_modes: int,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    计算GRPO损失
    
    Args:
        current_poses_cls: 当前策略的分类logits [bs, num_modes]
        ref_poses_cls: 参考策略的分类logits [bs, num_modes]
        rewards: PDM奖励 [bs, num_modes]
        num_modes: 轨迹模式数量
        clip_advantage_lower_quantile: 优势函数裁剪下分位数
        clip_advantage_upper_quantile: 优势函数裁剪上分位数
        clip_ratio: PPO裁剪比例
    """
    
    batch_size = current_poses_cls.shape[0]
    
    # 1. 计算优势函数
    mean_r = rewards.mean(dim=1, keepdim=True)  # [bs, 1]
    std_r = rewards.std(dim=1, keepdim=True) + 1e-8  # [bs, 1]
    #advantages = ((rewards - mean_r) / std_r)  # [bs, num_modes]
    
    # 确保std_r不为零，且没有NaN
    std_r = torch.where(std_r < 1e-6, torch.ones_like(std_r) * 1e-6, std_r)
    std_r = torch.clamp(std_r, min=1e-6, max=1e6)  # 防止过大过小
    
    advantages = (rewards - mean_r) / (std_r + 1e-8)  # [bs, num_modes]
    
    # 扁平化并裁剪优势函数
    advantages_flat = advantages.view(-1).detach()  # [bs * num_modes]
    advantages_flat = advantages_flat.float()
    adv_min = torch.quantile(advantages_flat, clip_advantage_lower_quantile)
    adv_max = torch.quantile(advantages_flat, clip_advantage_upper_quantile)
    advantages_clipped = advantages.clamp(min=adv_min, max=adv_max)
    
    # 2. 计算对数概率比
    current_probs = F.softmax(current_poses_cls, dim=-1)  # [bs, num_modes]
    ref_probs = F.softmax(ref_poses_cls, dim=-1)  # [bs, num_modes]
    
    # 避免数值问题
    current_probs = current_probs.clamp(min=1e-7, max=1.0)
    ref_probs = ref_probs.clamp(min=1e-7, max=1.0)
    
    log_ratios = torch.log(current_probs) - torch.log(ref_probs)
    
    policy_loss = -torch.mean(log_ratios * advantages)
    
    return policy_loss

def compute_grpo_loss2(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    num_modes: int,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    计算GRPO损失
    """
    batch_size = current_poses_cls.shape[0]
    
    # ===== 1. 输入清理（保持原样） =====
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
    current_poses_cls = torch.nan_to_num(current_poses_cls, nan=0.0)
    ref_poses_cls = torch.nan_to_num(ref_poses_cls, nan=0.0)
    
    # ===== 2. 安全计算优势函数（更合理的范围） =====
    # PDM奖励通常在[-1, 1]或[0, 1]范围
    # 使用更宽松的裁剪，只处理极端值
    rewards = torch.clamp(rewards, -2.0, 2.0)  # 宽松裁剪
    
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True)
    
    # 更合理的std范围
    min_std = 0.1  # 避免除零，同时保持数值稳定
    max_std = 2.0  # 允许一定的奖励方差
    
    std_r = torch.where(std_r < min_std, torch.ones_like(std_r) * min_std, std_r)
    std_r = torch.clamp(std_r, min=min_std, max=max_std)
    
    advantages = (rewards - mean_r) / (std_r + 1e-8)
    
    # advantages的合理范围：根据经验，3-5个标准差是合理的
    advantages = torch.clamp(advantages, -4.0, 4.0)
    
    # ===== 3. 使用原分位数裁剪（更稳定） =====
    advantages_flat = advantages.view(-1).detach()
    
    advantages_flat = advantages_flat.float()
    # 2. 清理可能的NaN/inf（虽然前面清理过，再加一层保险）
    advantages_flat = torch.nan_to_num(advantages_flat, nan=0.0)
    
    # 检查是否有足够的数据计算分位数
    if advantages_flat.numel() > 10:  # 至少有10个值
        try:
            adv_min = torch.quantile(advantages_flat, clip_advantage_lower_quantile)
            adv_max = torch.quantile(advantages_flat, clip_advantage_upper_quantile)
            advantages_clipped = advantages.clamp(min=adv_min, max=adv_max)
        except:
            # 分位数计算失败，使用原始advantages
            advantages_clipped = advantages
    else:
        advantages_clipped = advantages
    
    # ===== 4. 安全计算对数概率比 =====
    current_probs = F.softmax(current_poses_cls, dim=-1)
    ref_probs = F.softmax(ref_poses_cls, dim=-1)
    
    # 使用更合理的eps，避免影响正常训练
    eps = 1e-6  # 1e-6对softmax输出是安全的
    current_probs = current_probs.clamp(min=eps, max=1.0)
    ref_probs = ref_probs.clamp(min=eps, max=1.0)
    
    log_ratios = torch.log(current_probs) - torch.log(ref_probs)
    
    # PPO裁剪：保持你的clip_ratio参数
    if clip_ratio > 0:
        log_ratios = torch.clamp(log_ratios, -clip_ratio, clip_ratio)
    
    # ===== 5. 计算损失 =====
    # 使用advantages_clipped而不是原始advantages
    policy_loss = -torch.mean(log_ratios * advantages_clipped)
    
    # 检查loss是否合理
    max_loss = 10.0  # 合理的最大损失值
    if torch.abs(policy_loss) > max_loss:
        policy_loss = torch.clamp(policy_loss, -max_loss, max_loss)
    
    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        policy_loss = torch.tensor(0.0, device=current_poses_cls.device)
    
    return policy_loss

def compute_grpo_loss3(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    num_modes: int,
    clip_advantage_lower_quantile: float = 0.0,
    clip_advantage_upper_quantile: float = 1.0,
    clip_ratio: float = 0.2
) -> torch.Tensor:
    """
    GRPO loss with PPO-style probability ratio clipping.
    """
    rewards = torch.nan_to_num(rewards, nan=0.0).float()
    current_poses_cls = torch.nan_to_num(current_poses_cls, nan=0.0).float()
    ref_poses_cls = torch.nan_to_num(ref_poses_cls, nan=0.0).float()

    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp(min=0.01)
    advantages = ((rewards - mean_r) / std_r).detach()

    current_log_probs = F.log_softmax(current_poses_cls, dim=-1)
    ref_log_probs = F.log_softmax(ref_poses_cls.detach(), dim=-1)

    log_ratios = current_log_probs - ref_log_probs
    ratios = torch.exp(log_ratios)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    return policy_loss