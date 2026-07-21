"""Focused tests for the preregistered Stage-13 protocol."""

import copy
import hashlib
import subprocess

import pytest
import torch

from scripts.evaluation.check_grpo_stage13_gate import (
    BASE_CHECKPOINT,
    evaluate_multi,
    evaluate_single,
)
from scripts.training.check_grpo_stage13_checkpoint import (
    CURRENT,
    OLD,
    REFERENCE,
    SYNC_KEY,
    inspect_checkpoint,
)


def artifact(tokens, selected=0.5):
    records = [
        {
            "token": f"token-{index:05d}",
            "selector_logits_source": "reference",
            "selected_reward": selected,
            "candidate_reward": 0.4,
            "oracle_reward": 0.8,
            "selected_components": {
                "collision": 1.0,
                "drivable": 1.0,
                "progress": 0.7,
                "ttc": 1.0,
            },
        }
        for index in range(tokens)
    ]
    digest = hashlib.sha256(
        "\n".join(record["token"] for record in records).encode()
    ).hexdigest()
    return {
        "summary": {
            "checkpoint": BASE_CHECKPOINT,
            "reference_checkpoint": BASE_CHECKPOINT,
            "num_tokens": tokens,
            "requested_limit": tokens,
            "num_failures": 0,
            "completed": True,
            "selector_logits_source": "reference",
            "token_set_sha256": digest,
            "schedule": {
                "truncation_timestep": 8,
                "roll_timesteps": [8, 0],
                "scheduler_num_inference_steps": 125,
            },
        },
        "records": records,
    }


def shifted(base, delta, component_delta=0.0):
    result = copy.deepcopy(base)
    for record in result["records"]:
        record["selected_reward"] += delta
        for name in record["selected_components"]:
            record["selected_components"][name] += component_delta
    return result


def test_pilot_gate_uses_selected_pdms_and_all_components():
    base = artifact(3072)
    passed = evaluate_single(
        "pilot-dev-select", base, shifted(base, 0.0016), [], 100, 7
    )
    assert passed["passed"] is True

    unsafe = shifted(base, 0.0016)
    for record in unsafe["records"]:
        record["selected_components"]["progress"] -= 0.001
    failed = evaluate_single("pilot-dev-select", base, unsafe, [], 100, 7)
    assert failed["passed"] is False
    assert "progress delta is < 0" in failed["failures"]


def test_seed0_gate_requires_registered_kl_and_complete_identity():
    base = artifact(1024)
    candidate = shifted(base, 0.0021)
    assert evaluate_single(
        "seed0-fixed1024", base, candidate, [4e-4], 100, 11
    )["passed"]
    candidate["summary"]["completed"] = False
    failed = evaluate_single(
        "seed0-fixed1024", base, candidate, [6e-4], 100, 11
    )
    assert failed["passed"] is False
    assert "rolling generation KL is > 5e-4" in failed["failures"]


def test_three_seed_gate_and_result_classification():
    base = artifact(1024)
    candidates = [shifted(base, delta) for delta in (0.0021, 0.0022, 0.0023)]
    result = evaluate_multi(
        "three-seed-fixed1024", [base], candidates, [1e-4] * 3, 100, 13
    )
    assert result["passed"] is True
    assert result["result_classification"]["beats_existing_champion"] is True
    assert result["result_classification"]["strong_at_least_0.003"] is False


def checkpoint_payload(step=128):
    base_state = {
        CURRENT + "layers.0.ffn.weight": torch.zeros(2),
        CURRENT + "layers.1.ffn.weight": torch.zeros(2),
        CURRENT + "layers.0.task_decoder.plan_cls_branch.weight": torch.zeros(2),
        "agent._transfuser_model._backbone.weight": torch.zeros(2),
    }
    candidate_state = {key: value.clone() for key, value in base_state.items()}
    candidate_state[CURRENT + "layers.0.ffn.weight"] += 0.1
    candidate_state[CURRENT + "layers.1.ffn.weight"] += 0.1
    for key, value in base_state.items():
        if key.startswith(CURRENT):
            suffix = key[len(CURRENT):]
            candidate_state[REFERENCE + suffix] = value.clone()
            candidate_state[OLD + suffix] = value.clone()
    candidate_state[SYNC_KEY] = torch.tensor(((step - 1) // 32) * 32)
    return (
        {"state_dict": base_state},
        {
            "global_step": step,
            "optimizer_states": [{}],
            "lr_schedulers": [{}],
            "state_dict": candidate_state,
        },
    )


def test_checkpoint_audit_allows_synced_old_policy_but_not_reference_drift():
    base, candidate = checkpoint_payload()
    assert inspect_checkpoint(base, candidate, 128)["passed"] is True
    candidate["state_dict"][REFERENCE + "layers.0.ffn.weight"] += 0.1
    failed = inspect_checkpoint(base, candidate, 128)
    assert failed["passed"] is False
    assert "reference-policy tensors differ" in failed["failures"][-1]


def test_runner_rejects_unregistered_phase_before_hardware_access():
    result = subprocess.run(
        [
            "bash",
            "scripts/training/run_diffusiondrive_grpo_stage13.sh",
            "resume",
            "0",
            "stage13_bad",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "PHASE must be audit or formal" in result.stdout
