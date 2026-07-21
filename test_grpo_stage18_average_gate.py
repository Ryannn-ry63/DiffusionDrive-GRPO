import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import check_grpo_stage18_average_gate as stage18


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "artifacts/grpo_stage9/base_fixed1024.json"
PAIRED = ROOT / "artifacts/grpo_stage17/fixed1024/stage17.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(
        stage18,
        "bootstrap_ci",
        lambda values, seed=20260721, samples=10000: [float(np.mean(values)), float(np.mean(values))],
    )


def test_stage18_fixed_diagnostic_uses_exact_paired_noise(monkeypatch):
    fast_bootstrap(monkeypatch)
    result = stage18.evaluate_gate(load(ORIGINAL), load(PAIRED), "fixed-diagnostic")
    metrics = result["metrics"]
    assert result["passed"]
    assert result["decision_scope"] == "diagnostic_only"
    assert metrics["d_minus_b"] == pytest.approx(0.021662254497641698)
    assert metrics["d_minus_c"] == pytest.approx(0.0019324200984556228)
    assert metrics["fallback_count"] == 185
    assert metrics["catastrophic_admitted_count"] == 3
    assert metrics["worst_d_minus_b"] == pytest.approx(-0.8297449946403503)
    assert result["tail_metrics_are_nonblocking"] is True


def test_stage18_rejects_mixed_verifier_provenance():
    original, paired = load(ORIGINAL), load(PAIRED)
    paired["summary"] = {
        **paired["summary"],
        "paired_risk": {**paired["summary"]["paired_risk"], "checkpoint_sha256": "wrong"},
    }
    with pytest.raises(RuntimeError, match="verifier checkpoint SHA mismatch"):
        stage18.validate_and_extract(original, paired, 1024)


def test_stage18_rejects_wrong_feature_cache_provenance():
    original, paired = load(ORIGINAL), load(PAIRED)
    paired["summary"] = {**paired["summary"], "cache_path": "/tmp/not-the-locked-cache"}
    with pytest.raises(RuntimeError, match="feature cache provenance mismatch"):
        stage18.validate_and_extract(original, paired, 1024)


def test_stage18_rejects_selected_reward_from_wrong_branch():
    original, paired = load(ORIGINAL), load(PAIRED)
    first = {**paired["records"][0], "selected_reward": paired["records"][0]["selected_reward"] + 0.1}
    paired["records"] = [first, *paired["records"][1:]]
    with pytest.raises(RuntimeError, match="selected reward branch"):
        stage18.validate_and_extract(original, paired, 1024)


def test_stage18_dev_partitions_are_pairwise_disjoint():
    paths = (
        ROOT / "artifacts/grpo_stage9/base_fixed1024.json",
        ROOT / "artifacts/grpo_stage0/dev_select3072_manifest.json",
        ROOT / "artifacts/grpo_stage0/dev_confirm1024_manifest.json",
    )
    token_sets = [
        {str(record["token"]) for record in load(path)["records"]}
        for path in paths
    ]
    assert [len(tokens) for tokens in token_sets] == [1024, 3072, 1024]
    assert not token_sets[0] & token_sets[1]
    assert not token_sets[0] & token_sets[2]
    assert not token_sets[1] & token_sets[2]


def test_stage18_dev_select_gate_ignores_tail_but_enforces_mean(monkeypatch):
    tokens = [f"token-{index}" for index in range(3072)]
    arrays = {
        "a": np.full(3072, 0.70),
        "b": np.full(3072, 0.70),
        "c": np.full(3072, 0.725),
        "d": np.full(3072, 0.725),
        "fallback": np.zeros(3072, dtype=bool),
        "base_components": {name: np.ones(3072) for name in stage18.COMPONENTS},
        "selected_components": {name: np.ones(3072) for name in stage18.COMPONENTS},
    }
    arrays["d"][0] = -0.1
    arrays["c"][0] = -0.1
    monkeypatch.setattr(stage18, "validate_and_extract", lambda *args: (tokens, arrays))
    fast_bootstrap(monkeypatch)
    result = stage18.evaluate_gate({}, {}, "dev-select")
    assert result["metrics"]["worst_d_minus_b"] < -0.5
    assert result["passed"]

    arrays["d"][:] = 0.71
    arrays["c"][:] = 0.71
    result = stage18.evaluate_gate({}, {}, "dev-select")
    assert not result["passed"]
    assert "D-B mean/CI gate failed" in result["failures"]
