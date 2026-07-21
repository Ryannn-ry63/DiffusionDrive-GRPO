from dataclasses import replace

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


def test_generation_ablation_with_save_all_does_not_monitor_val_reward():
    agent = object.__new__(TransfuserAgent)
    agent._config = replace(
        TransfuserConfig(),
        grpo_training_mode="generation",
        grpo_checkpoint_save_top_k=-1,
        grpo_checkpoint_every_n_train_steps=0,
    )

    callbacks = agent.get_training_callbacks()

    checkpoint = callbacks[1]
    assert checkpoint.monitor is None
    assert checkpoint.save_top_k == -1
    assert checkpoint.every_n_epochs == 1


def test_generation_formal_keeps_registered_step_checkpoint():
    agent = object.__new__(TransfuserAgent)
    agent._config = replace(
        TransfuserConfig(),
        grpo_training_mode="generation",
        grpo_checkpoint_save_top_k=-1,
        grpo_checkpoint_every_n_train_steps=64,
    )

    callbacks = agent.get_training_callbacks()

    assert len(callbacks) == 3
    assert callbacks[2]._every_n_train_steps == 64
    assert callbacks[2].monitor is None
