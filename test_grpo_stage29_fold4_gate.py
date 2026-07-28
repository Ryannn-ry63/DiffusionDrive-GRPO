from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluation import check_grpo_stage29_fold4 as gate


N = 2042
HARD = 400


def make_summary(monkeypatch, *, selectable=True, selected=None, candidate=None,
                 oracle=None, component=-0.0, ci=(0.001, 0.008), noises=None):
    monkeypatch.setattr(gate, "bootstrap_whole_log", lambda values, logs: list(ci))
    if selected is None:
        selected = np.r_[np.full(HARD, 0.02), np.full(N - HARD, 0.001)]
    selected = np.asarray(selected, dtype=np.float64)
    if candidate is None:
        candidate = np.full(N, 0.001)
    if oracle is None:
        oracle = np.zeros(N)
    base = np.r_[np.full(HARD, 0.7), np.full(N - HARD, 0.9)]
    if noises is None:
        noises = {
            gate.NOISES[0]: selected[:1021].tolist(),
            gate.NOISES[1]: selected[1021:].tolist(),
        }
    components = {name: np.full(N, component).tolist() for name in gate.COMPONENTS}
    return gate.summarize(
        "H1", {"branch": "H", "epoch": 1, "selectable": selectable},
        selected, np.asarray(candidate), np.asarray(oracle), np.maximum(selected, 0),
        base, [f"log_{i % 151}" for i in range(N)], noises, components,
    )


def test_stage29_fold4_full_numeric_gate_and_control_lock(monkeypatch):
    result = make_summary(monkeypatch)
    assert result["eligible"] and result["passes_numeric_gate"]
    assert not result["strong_gain_at_least_0.005"]
    control = make_summary(monkeypatch, selectable=False)
    assert control["passes_numeric_gate"] and not control["eligible"]


def test_stage29_fold4_ci_and_namespace_fail_closed(monkeypatch):
    assert not make_summary(monkeypatch, ci=(-1e-4, 0.01))["checks"][
        "whole_log_ci_lower_strictly_positive"
    ]
    selected = np.r_[np.full(1021, 0.01), np.full(1021, -0.001)]
    noises = {gate.NOISES[0]: selected[:1021], gate.NOISES[1]: selected[1021:]}
    result = make_summary(monkeypatch, selected=selected, noises=noises)
    assert not result["checks"]["both_namespace_means_strictly_positive"]


def test_stage29_fold4_trim_and_win_gate(monkeypatch):
    selected = np.r_[np.full(700, 0.003), np.full(N - 700, -0.002)]
    result = make_summary(monkeypatch, selected=selected)
    assert not result["checks"]["wins_exceed_losses_excluding_exact_ties"]
    assert result["trimmed10_mean_delta"] < 0
    assert not result["checks"]["trimmed10_mean_nonnegative"]


def test_stage29_fold4_hard_and_mature_gates(monkeypatch):
    hard_fail = np.r_[np.full(HARD, 0.009), np.full(N - HARD, 0.001)]
    assert not make_summary(monkeypatch, selected=hard_fail)["checks"][
        "hard_scene_gain_at_least_0.010"
    ]
    mature_fail = np.r_[np.full(HARD, 0.03), np.full(N - HARD, -0.001)]
    assert not make_summary(monkeypatch, selected=mature_fail)["checks"][
        "mature_scene_gain_at_least_negative_0.0002"
    ]


def test_stage29_fold4_candidate_and_oracle_gates(monkeypatch):
    assert not make_summary(monkeypatch, candidate=np.zeros(N))["checks"][
        "candidate_mean_delta_strictly_positive"
    ]
    assert not make_summary(monkeypatch, oracle=np.full(N, -0.0006))["checks"][
        "oracle_candidate_delta_at_least_negative_0.0005"
    ]


def test_stage29_fold4_component_and_catastrophe_gates(monkeypatch):
    assert not make_summary(monkeypatch, component=-0.0006)["checks"][
        "all_components_at_least_negative_0.0005"
    ]
    selected = np.r_[np.full(HARD, 0.02), np.full(N - HARD, 0.001)]
    selected[0] = -0.5
    assert not make_summary(monkeypatch, selected=selected)["checks"][
        "catastrophic_count_zero"
    ]


def test_stage29_fold4_pooled_threshold_and_strong_flag(monkeypatch):
    selected = np.r_[np.full(HARD, 0.012), np.full(N - HARD, 0.0001)]
    result = make_summary(monkeypatch, selected=selected)
    assert not result["checks"]["pooled_gain_at_least_0.003"]
    strong = np.r_[np.full(HARD, 0.03), np.full(N - HARD, 0.001)]
    assert make_summary(monkeypatch, selected=strong)["strong_gain_at_least_0.005"]


def test_stage29_fold4_manifest_provenance_sha_fails_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="manifest SHA drifted"):
        gate.load_manifest(manifest, "0" * 64)
