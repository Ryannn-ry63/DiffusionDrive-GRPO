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
    Three-part loss:
      1. GT-matched regression  (imitation baseline, keeps trajectory quality)
      2. Reward-directed classification  (RL signal – teaches mode selector to
         favour high-PDM-score modes, enabling improvement beyond human GT)
      3. KL regularisation  (prevents catastrophic drift from pretrained policy)

    During validation (no rewards), only regression loss is returned so that
    checkpoint quality can be monitored.
    """

    device = predictions["trajectory"].device
    zero = torch.tensor(0.0, device=device)
    tau = getattr(config, "reward_temperature", 0.1)

    # ---- 1. GT-matched regression (always) ----
    reg_loss = config.trajectory_reg_weight * F.l1_loss(
        predictions["trajectory"], targets["trajectory"]
    )

    # Validation path
    if "rewards" not in predictions or predictions["rewards"] is None:
        return {
            "loss": reg_loss,
            "reg_loss": reg_loss,
            "reward_cls_loss": zero,
            "reward_reg_loss": zero,
            "kl_loss": zero,
        }

    # ---- Training path (rewards available) ----
    rewards = predictions["rewards"].float()            # [bs, num_modes]
    poses_cls = predictions["final_poses_cls"].float()  # [bs, num_modes]

    # ---- 2. Reward-directed classification ----
    # Convert PDM rewards into a soft target distribution; the classifier
    # learns to assign high probability to modes with high driving scores.
    reward_target = F.softmax(rewards / tau, dim=-1).detach()
    log_probs = F.log_softmax(poses_cls, dim=-1)
    reward_cls_loss = config.trajectory_cls_weight * (
        -torch.mean(torch.sum(reward_target * log_probs, dim=-1))
    )

    # ---- 3. Reward-weighted all-mode regression ----
    # High-reward modes also receive regression supervision so that their
    # trajectory outputs stay accurate.  This complements (1) which only
    # trains the single GT-closest mode.
    reward_reg_loss = zero
    final_poses_reg = predictions.get("final_poses_reg")
    if final_poses_reg is not None:
        gt_traj = targets["trajectory"]                        # [bs, 8, 3]
        gt_expanded = gt_traj.unsqueeze(1).expand_as(final_poses_reg)
        per_mode_l1 = F.l1_loss(
            final_poses_reg, gt_expanded, reduction="none"
        ).mean(dim=(-1, -2))                                   # [bs, num_modes]
        reward_weights = F.softmax(rewards / tau, dim=-1).detach()
        reward_reg_loss = config.trajectory_reg_weight * (
            (reward_weights * per_mode_l1).sum(dim=-1).mean()
        )

    # ---- 4. KL regularisation ----
    kl_loss_raw = predictions.get("kl_div", zero)
    weighted_kl = config.kl_loss_weight * kl_loss_raw

    total_loss = reg_loss + reward_cls_loss + reward_reg_loss + weighted_kl

    return {
        "loss": total_loss,
        "reg_loss": reg_loss,
        "reward_cls_loss": reward_cls_loss,
        "reward_reg_loss": reward_reg_loss,
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
