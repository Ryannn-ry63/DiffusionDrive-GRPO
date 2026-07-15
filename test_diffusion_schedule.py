"""Stage-0 checks for the explicit truncated-DDIM schedule."""

from dataclasses import replace

import pytest

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead


def _build_head(config: TransfuserConfig) -> TrajectoryHead:
    return TrajectoryHead(
        num_poses=config.trajectory_sampling.num_poses,
        d_ffn=config.tf_d_ffn,
        d_model=config.tf_d_model,
        plan_anchor_path=config.plan_anchor_path,
        config=config,
    )


@pytest.mark.parametrize(
    ("truncation_timestep", "roll_timesteps", "num_inference_steps", "expected"),
    [
        (8, (10, 0), 1000, ((10, 9), (0, -1))),
        (10, (10, 0), 100, ((10, 0), (0, -10))),
        (8, (8, 0), 125, ((8, 0), (0, -8))),
    ],
)
def test_schedule_metadata(
    truncation_timestep, roll_timesteps, num_inference_steps, expected
):
    config = replace(
        TransfuserConfig(),
        metric_cache_path="",
        diffusion_truncation_timestep=truncation_timestep,
        diffusion_roll_timesteps=roll_timesteps,
        diffusion_scheduler_num_inference_steps=num_inference_steps,
    )
    schedule = _build_head(config).get_roll_schedule()

    assert schedule["truncation_timestep"] == truncation_timestep
    assert schedule["roll_timesteps"] == roll_timesteps
    assert schedule["transitions"] == expected


def test_schedule_rejects_non_decreasing_timesteps():
    config = replace(
        TransfuserConfig(),
        metric_cache_path="",
        diffusion_roll_timesteps=(0, 10),
    )
    with pytest.raises(ValueError, match="strictly decreasing"):
        _build_head(config)


def test_default_schedule_is_audited_and_aligned():
    config = replace(TransfuserConfig(), metric_cache_path="")
    schedule = _build_head(config).get_roll_schedule()

    assert schedule["truncation_timestep"] == schedule["roll_timesteps"][0] == 8
    assert schedule["transitions"] == ((8, 0), (0, -8))
