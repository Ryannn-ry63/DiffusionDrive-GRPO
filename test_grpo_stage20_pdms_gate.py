import copy
import hashlib
import json

import pytest

from scripts.evaluation import check_grpo_stage20_gate as gate


def _sha(tokens):
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def _artifact(tokens, label, noise, candidate_sha, reward, component=1.0):
    schedule = gate.SHORT_SCHEDULE if label == "A" else gate.FULL_SCHEDULE
    checkpoint_sha = gate.BASE_SHA256 if label in {"A", "B"} else candidate_sha
    return {
        "summary": {
            "checkpoint_sha256": checkpoint_sha,
            "reference_checkpoint_sha256": gate.BASE_SHA256,
            "generation_policy_algorithm": (
                "legacy_ppo" if label == "A" else "diffgrpo_selected_anchor"
            ),
            "evaluation_noise_namespace": noise,
            "selector_logits_source": "reference",
            "schedule": schedule,
            "log_split": "val",
            "completed": True,
            "num_failures": 0,
            "num_tokens": len(tokens),
            "token_set_sha256": _sha(tokens),
        },
        "records": [
            {
                "token": token,
                "selected_reward": reward,
                "selected_components": {
                    name: component for name in gate.COMPONENTS
                },
            }
            for token in tokens
        ],
    }


def _case(tmp_path, monkeypatch, c_reward=0.51):
    tokens = [f"token-{index:04d}" for index in range(1000)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"token": token, "log_name": f"log-{index // 5:04d}"}
                    for index, token in enumerate(tokens)
                ]
            }
        )
    )
    monkeypatch.setitem(
        gate.EXPECTED,
        "dev-select",
        {"count": len(tokens), "token_sha": _sha(tokens), "cb_min": 0.008},
    )
    candidate_sha = "c" * 64
    paths = {label: [] for label in "ABC"}
    rewards = {"A": 0.49, "B": 0.50, "C": c_reward}
    for label in "ABC":
        for noise in gate.NOISES:
            path = tmp_path / f"{label}-{noise}.json"
            path.write_text(
                json.dumps(
                    _artifact(
                        tokens, label, noise, candidate_sha, rewards[label]
                    )
                )
            )
            paths[label].append(path)
    return manifest, paths, candidate_sha


def test_stage20_pdms_gate_passes_registered_gain(tmp_path, monkeypatch):
    manifest, paths, candidate_sha = _case(tmp_path, monkeypatch)
    result = gate.evaluate_gate(
        gate="dev-select",
        manifest=manifest,
        cell_paths=paths,
        expected_candidate_sha=candidate_sha,
        bootstrap_samples=20,
    )
    assert result["passed"]
    assert result["metrics"]["two_noise_c_minus_b"] == pytest.approx(0.01)
    assert result["metrics"]["catastrophic_count"] == 0


def test_stage20_pdms_gate_rejects_nonpositive_gain(tmp_path, monkeypatch):
    manifest, paths, candidate_sha = _case(
        tmp_path, monkeypatch, c_reward=0.499
    )
    result = gate.evaluate_gate(
        gate="dev-select",
        manifest=manifest,
        cell_paths=paths,
        expected_candidate_sha=candidate_sha,
        bootstrap_samples=20,
    )
    assert not result["passed"]
    assert any("non-positive" in failure for failure in result["failures"])


def test_stage20_pdms_gate_fails_closed_on_provenance(tmp_path, monkeypatch):
    manifest, paths, candidate_sha = _case(tmp_path, monkeypatch)
    payload = json.loads(paths["C"][0].read_text())
    payload["summary"]["selector_logits_source"] = "current"
    paths["C"][0].write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="provenance"):
        gate.evaluate_gate(
            gate="dev-select",
            manifest=manifest,
            cell_paths=paths,
            expected_candidate_sha=candidate_sha,
            bootstrap_samples=20,
        )
