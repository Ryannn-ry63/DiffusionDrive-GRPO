from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex


def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Combined loss: L1 regression (imitation) + GRPO policy gradient + KL regularization.
    Regression loss is always present to prevent trajectory quality degradation.
    GRPO + KL are only added during training when rewards are available.
    """

    device = predictions["trajectory"].device

    # --- Regression loss (always computed) ---
    reg_loss = config.trajectory_reg_weight * F.l1_loss(
        predictions["trajectory"], targets["trajectory"]
    )

    # Validation: no rewards available, return regression loss only
    if "rewards" not in predictions or predictions["rewards"] is None:
        return {
            "loss": reg_loss,
            "reg_loss": reg_loss,
            "grpo_loss": torch.tensor(0.0, device=device),
            "kl_loss": torch.tensor(0.0, device=device),
        }

    # --- Training: regression + GRPO + KL ---
    grpo_loss_raw = compute_grpo_loss_clipped(
        current_poses_cls=predictions["final_poses_cls"],
        ref_poses_cls=predictions["final_ref_poses_cls"],
        rewards=predictions["rewards"],
        clip_ratio=0.2,
    )

    kl_loss_raw = predictions.get("kl_div", torch.tensor(0.0, device=device))

    weighted_grpo = config.policy_loss_weight * grpo_loss_raw
    weighted_kl = config.kl_loss_weight * kl_loss_raw

    total_loss = reg_loss + weighted_grpo + weighted_kl

    return {
        "loss": total_loss,
        "reg_loss": reg_loss,
        "grpo_loss": weighted_grpo,
        "kl_loss": weighted_kl,
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

def compute_grpo_loss_clipped(
    current_poses_cls: torch.Tensor,
    ref_poses_cls: torch.Tensor,
    rewards: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """
    GRPO loss with PPO-style clipping on the probability ratio.

    Args:
        current_poses_cls: current policy logits  [bs, num_modes]
        ref_poses_cls:     reference policy logits [bs, num_modes]
        rewards:           per-mode PDM rewards    [bs, num_modes]
        clip_ratio:        PPO clip epsilon
    Returns:
        scalar policy loss
    """
    rewards = torch.nan_to_num(rewards, nan=0.0).float()
    current_poses_cls = torch.nan_to_num(current_poses_cls, nan=0.0).float()
    ref_poses_cls = torch.nan_to_num(ref_poses_cls, nan=0.0).float()

    # --- advantages (per-sample normalisation) ---
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True).clamp(min=0.01)
    advantages = ((rewards - mean_r) / std_r).detach()
    advantages = advantages.clamp(-5.0, 5.0)

    # --- probability ratio with PPO clipping ---
    eps = 1e-8
    current_probs = F.softmax(current_poses_cls, dim=-1).clamp(min=eps)
    ref_probs = F.softmax(ref_poses_cls, dim=-1).clamp(min=eps)

    ratio = current_probs / ref_probs                                    # π_θ / π_ref
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        policy_loss = torch.tensor(0.0, device=current_poses_cls.device)

    return policy_loss