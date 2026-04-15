from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex


# TODO: 通读一下，其实compute loss被调用了两次：一次在_trajectory_head，其实是对每一个轨迹，都计算了loss，但没有真正用于梯度回传；
# 一次是在navsim/planning/training/agent_lightning_module.py L32里面，最终计算得到的loss dict才会用于下游计算。
# TODO: 在 _trajectory_head 里的compute loss可以注释了，其实不对下游训练产生影响；记得把计算grpo loss的逻辑，从_trajectory_head 的forward中，搬到这里来
# 记住我们只finetune _trajectory_head 

def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Combined loss: IL regression (anchors trajectory quality) + GRPO (nudges mode selection) + KL.
    """
    device = predictions["trajectory"].device
    reg_weight = getattr(config, 'trajectory_reg_weight', 8.0)
    grpo_weight = getattr(config, 'policy_loss_weight', 0.1)
    kl_weight = getattr(config, 'kl_loss_weight', 0.1)

    # IL regression loss: always computed, protects trajectory quality through shared features
    reg_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])

    if "rewards" not in predictions or predictions["rewards"] is None:
        # Validation: only regression loss (now we have a real validation signal)
        return {
            "loss": reg_weight * reg_loss,
            "grpo_loss": torch.tensor(0.0, device=device),
            "kl_loss": torch.tensor(0.0, device=device),
            "reg_loss": reg_weight * reg_loss,
        }

    # GRPO policy gradient loss
    grpo_loss = compute_grpo_loss3(
        current_poses_cls=predictions["final_poses_cls"],
        ref_poses_cls=predictions["final_ref_poses_cls"],
        rewards=predictions["rewards"],
        num_modes=predictions.get("num_modes", 20),
        clip_ratio=0.2,
    )

    kl_loss = predictions.get("kl_div", torch.tensor(0.0, device=device))

    total_loss = reg_weight * reg_loss + grpo_weight * grpo_loss + kl_weight * kl_loss

    return {
        'loss': total_loss,
        'grpo_loss': grpo_weight * grpo_loss,
        'kl_loss': kl_weight * kl_loss,
        'reg_loss': reg_weight * reg_loss,
    }


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