"""Independent public/challenger sampling and full-chain replay for Stage39."""

from __future__ import annotations

from typing import Dict

import torch

from navsim.agents.diffusiondrive.diffusion_grpo import (
    _replay_full_chain_actions,
    _replay_reference_mean_kl,
    _sample_full_chain_actions,
)


def _validate_stage39_head(head, group_size: int) -> None:
    if int(group_size) != 8:
        raise ValueError("Stage39 requires exactly eight rollout groups")
    if int(head.plan_anchor.shape[0]) != 20:
        raise RuntimeError("Stage39 requires the complete public 20-anchor bank")
    if getattr(head, "stage39_challenger_decoder", None) is None:
        raise RuntimeError("Stage39 challenger decoder is missing")
    public_parameters = tuple(head.diff_decoder.parameters())
    challenger_parameters = tuple(head.stage39_challenger_decoder.parameters())
    if len(public_parameters) != len(challenger_parameters):
        raise RuntimeError("Stage39 public/challenger decoder structures differ")
    if any(
        public.data_ptr() == challenger.data_ptr()
        for public, challenger in zip(public_parameters, challenger_parameters)
    ):
        raise RuntimeError("Stage39 public/challenger parameters alias storage")


def collect_stage39_sampling_trace(
    head,
    group_size: int,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
) -> Dict[str, object]:
    """Sample independent public and challenger banks.

    Initial, transition and final noise are deliberately independent.  This is
    candidate-support expansion, not a common-random-number paired comparison.
    """
    _validate_stage39_head(head, group_size)
    batch = ego_query.shape[0]
    groups = int(group_size)
    modes = 20
    clean = head.norm_odo(
        head.plan_anchor.unsqueeze(0).unsqueeze(1)
        .expand(batch, groups, -1, -1, -1)
        .reshape(batch, groups * modes, *head.plan_anchor.shape[1:])
    )
    timesteps = torch.full(
        (batch,),
        int(head._truncation_timestep),
        device=clean.device,
        dtype=torch.long,
    )
    challenger_initial_noise = torch.randn_like(clean)
    public_initial_noise = torch.randn_like(clean)
    challenger_initial = head.diffusion_scheduler.add_noise(
        original_samples=clean,
        noise=challenger_initial_noise,
        timesteps=timesteps,
    ).detach()
    public_initial = head.diffusion_scheduler.add_noise(
        original_samples=clean,
        noise=public_initial_noise,
        timesteps=timesteps,
    ).detach()
    (
        challenger_states,
        challenger_actions,
        challenger_final,
        challenger_logits,
    ) = _sample_full_chain_actions(
        head,
        head.stage39_challenger_decoder,
        challenger_initial,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    with torch.no_grad():
        (
            public_states,
            public_actions,
            public_final,
            public_logits,
        ) = _sample_full_chain_actions(
            head,
            head.diff_decoder,
            public_initial,
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            global_img,
        )
    initial_noise_cosine = torch.nn.functional.cosine_similarity(
        challenger_initial_noise.flatten(1),
        public_initial_noise.flatten(1),
        dim=1,
    ).abs().mean()
    return {
        "challenger_states": challenger_states,
        "challenger_actions": challenger_actions,
        "challenger_final": challenger_final,
        "challenger_cls": challenger_logits,
        "public_states": public_states,
        "public_actions": public_actions,
        "public_final": public_final,
        "public_cls": public_logits,
        "group_size": clean.new_tensor(float(groups)),
        "sampled_chain_count": clean.new_tensor(float(2 * groups * modes)),
        "independent_initial_noise": (
            challenger_initial_noise.data_ptr() != public_initial_noise.data_ptr()
        ),
        "initial_noise_abs_cosine": initial_noise_cosine.detach(),
    }


def finalize_stage39_replay(
    head,
    trace: Dict[str, object],
    *,
    ego_query: torch.Tensor,
    agents_query: torch.Tensor,
    bev_feature: torch.Tensor,
    bev_spatial_shape,
    status_encoding: torch.Tensor,
    global_img: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    """Replay all 8×20 chains so every ablation has the same compute budget."""
    batch = ego_query.shape[0]
    groups, modes = 8, 20
    challenger_log_probs, _ = _replay_full_chain_actions(
        head,
        head.stage39_challenger_decoder,
        trace["challenger_states"],
        trace["challenger_actions"],
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    challenger_kl = _replay_reference_mean_kl(
        head,
        head.stage39_challenger_decoder,
        head.diff_decoder,
        trace["challenger_states"],
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    public_bc_log_probs, _ = _replay_full_chain_actions(
        head,
        head.stage39_challenger_decoder,
        trace["public_states"],
        trace["public_actions"],
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
    )
    steps = challenger_log_probs.shape[-1]
    expected_rows = batch * groups * modes
    if challenger_log_probs.shape[0] != batch or challenger_log_probs.shape[1] != groups * modes:
        raise RuntimeError(
            f"Stage39 challenger replay shape drifted; expected {expected_rows} chains"
        )
    shape = (batch, groups, modes, steps)
    return {
        "stage39_challenger_log_probs": challenger_log_probs.reshape(shape),
        "stage39_challenger_reference_kl": challenger_kl.reshape(shape),
        "stage39_public_bc_log_probs": public_bc_log_probs.reshape(shape),
        "stage39_sampled_chain_count": trace["sampled_chain_count"],
        "stage39_replayed_chain_count": challenger_log_probs.new_tensor(
            float(2 * groups * modes)
        ),
        "stage39_independent_initial_noise": challenger_log_probs.new_tensor(
            float(bool(trace["independent_initial_noise"]))
        ),
        "stage39_initial_noise_abs_cosine": trace[
            "initial_noise_abs_cosine"
        ].to(challenger_log_probs),
        "diffgrpo_num_denoising_steps": challenger_log_probs.new_tensor(
            float(steps)
        ),
    }

