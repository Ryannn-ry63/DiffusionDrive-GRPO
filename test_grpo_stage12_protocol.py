"""Focused tests for the Stage-12 continuation protocol and gates."""

import copy

import pytest
import torch
from scripts.evaluation.build_grpo_stage12_navtest_manifest import load_valid_tokens

from scripts.evaluation.check_grpo_stage12_gate import (
    DIAGNOSTIC_STEPS,
    evaluate_diagnostic,
    evaluate_multi,
    evaluate_seed0,
)
from scripts.training.check_grpo_stage12_resume import (
    CONTROLLER_SUFFIXES,
    inspect_resume_payload,
)


def artifact(
    *,
    tokens=1024,
    selected=0.5,
    candidate=0.4,
    oracle=0.8,
    components=None,
    source="reference",
):
    components = components or {
        "collision": 1.0,
        "drivable": 1.0,
        "progress": 0.7,
        "ttc": 1.0,
    }
    records = []
    for index in range(tokens):
        records.append(
            {
                "token": f"token-{index:05d}",
                "selected_reward": selected,
                "candidate_reward": candidate,
                "oracle_reward": oracle,
                "selected_components": dict(components),
            }
        )
    return {
        "summary": {
            "num_tokens": tokens,
            "token_set_sha256": f"sha-{tokens}",
            "selector_logits_source": source,
            "num_failures": 0,
        },
        "records": records,
    }


def shifted(base, *, selected=0.0, candidate=0.0, oracle=0.0, components=None):
    result = copy.deepcopy(base)
    components = components or {}
    for record in result["records"]:
        record["selected_reward"] += selected
        record["candidate_reward"] += candidate
        record["oracle_reward"] += oracle
        for name, delta in components.items():
            record["selected_components"][name] += delta
    return result


def test_stage12_diagnostic_requires_registered_curve_and_headroom():
    base = artifact()
    u128 = shifted(base, selected=0.0024, candidate=0.0012, oracle=0.0005)
    candidates = [
        shifted(base, selected=0.0028, candidate=0.0013, oracle=0.0005),
        shifted(base, selected=0.0032, candidate=0.0014, oracle=0.0005),
        shifted(base, selected=0.0029, candidate=0.0014, oracle=0.0005),
        shifted(base, selected=0.0029, candidate=0.0014, oracle=0.0005),
    ]

    result = evaluate_diagnostic(
        base, u128, candidates, DIAGNOSTIC_STEPS, samples=100, seed=7
    )

    assert result["passed"] is True
    assert result["eligible_stage9_steps"] == [512]
    with pytest.raises(ValueError, match="128, 512, 1024, 2048"):
        evaluate_diagnostic(base, u128, candidates[:2], (128, 512), 100, 7)


def test_stage12_diagnostic_stops_without_continuation_evidence():
    base = artifact()
    u128 = shifted(base, selected=0.0024, candidate=0.0012, oracle=0.0005)
    candidates = [
        shifted(base, selected=0.0025, candidate=0.0011, oracle=0.0005)
        for _ in DIAGNOSTIC_STEPS
    ]

    result = evaluate_diagnostic(
        base, u128, candidates, DIAGNOSTIC_STEPS, samples=100, seed=9
    )

    assert result["passed"] is False
    assert result["eligible_stage9_steps"] == []
    assert "no Stage-9 U512+" in result["failures"][-1]


def test_stage12_seed0_gate_checks_base_u128_gain_safety_and_kl():
    base = artifact()
    u128 = shifted(base, selected=0.0024, candidate=0.0012, oracle=0.0005)
    u256 = shifted(base, selected=0.0032, candidate=0.0014, oracle=0.0008)

    passed = evaluate_seed0(base, u128, u256, [1e-4], samples=100, seed=11)
    assert passed["passed"] is True
    assert passed["versus_stage10_u128"]["selected_mean"] == pytest.approx(0.0008)

    failed = evaluate_seed0(base, u128, u256, [3e-4], samples=100, seed=11)
    assert failed["passed"] is False
    assert "rolling KL is > 2.5e-4" in failed["failures"]


def test_stage12_seed0_gate_rejects_current_selector_artifact():
    base = artifact()
    u128 = shifted(base, selected=0.0024, candidate=0.0012, oracle=0.0005)
    u256 = shifted(base, selected=0.0032, candidate=0.0014, oracle=0.0008)
    u256["summary"]["selector_logits_source"] = "current"

    result = evaluate_seed0(base, u128, u256, [1e-4], samples=100, seed=13)

    assert result["passed"] is False
    assert "Stage-12 U256 selector source is not reference" in result["failures"]


def test_stage12_three_seed_gate_requires_positive_safe_reproduction():
    base = artifact()
    candidates = [
        shifted(base, selected=value, candidate=0.001, oracle=0.0005)
        for value in (0.0031, 0.0032, 0.0033)
    ]

    passed = evaluate_multi(
        "three-seed-fixed1024",
        [base],
        candidates,
        [1e-4, 1.1e-4, 1.2e-4],
        samples=100,
        seed=17,
    )
    assert passed["passed"] is True
    assert passed["mean_selected_difference"] == pytest.approx(0.0032)

    unsafe = copy.deepcopy(candidates)
    unsafe[0] = shifted(
        base,
        selected=0.0031,
        candidate=0.001,
        oracle=0.0005,
        components={"collision": -0.003},
    )
    failed = evaluate_multi(
        "three-seed-fixed1024",
        [base],
        unsafe,
        [1e-4, 1.1e-4, 1.2e-4],
        samples=100,
        seed=17,
    )
    assert failed["passed"] is False
    assert "mean collision delta is < 0" in failed["failures"]


def resume_payload(global_step=128, *, should_stop=False):
    state_dict = {}
    values = {
        "coefficient": torch.tensor(0.1),
        "rolling_values": torch.zeros(32),
        "rolling_index": torch.tensor(0),
        "rolling_count": torch.tensor(32),
        "observation_count": torch.tensor(global_step),
        "consecutive_hard_violations": torch.tensor(0),
        "rolling_mean": torch.tensor(3e-5),
        "should_stop": torch.tensor(should_stop),
    }
    for suffix in CONTROLLER_SUFFIXES:
        state_dict[f"agent._adaptive_kl_controller.{suffix}"] = values[suffix]
    return {
        "global_step": global_step,
        "optimizer_states": [{}],
        "lr_schedulers": [{}],
        "state_dict": state_dict,
    }


def test_stage12_resume_preflight_requires_complete_live_trainer_state():
    passed = inspect_resume_payload(resume_payload(), expected_global_step=128)
    assert passed["passed"] is True
    assert passed["adaptive_kl_should_stop"] is False

    incomplete = resume_payload(global_step=127, should_stop=True)
    incomplete["lr_schedulers"] = []
    incomplete["state_dict"].pop("agent._adaptive_kl_controller.rolling_values")
    failed = inspect_resume_payload(incomplete, expected_global_step=128)
    assert failed["passed"] is False
    assert len(failed["failures"]) == 4


def test_stage12_navtest_manifest_requires_complete_valid_unique_csv(tmp_path):
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text(
        ",token,valid,score\n0,token-a,True,0.5\n1,token-b,True,0.7\n2,average,True,0.6\n",
        encoding="utf-8",
    )
    assert load_valid_tokens(score_csv, expected_tokens=2) == ["token-a", "token-b"]

    score_csv.write_text(
        ",token,valid,score\n0,token-a,True,0.5\n1,token-a,True,0.7\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_valid_tokens(score_csv, expected_tokens=2)
