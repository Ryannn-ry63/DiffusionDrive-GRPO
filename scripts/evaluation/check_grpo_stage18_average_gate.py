#!/usr/bin/env python3
"""Apply the pre-registered Stage-18 average-primary gate.

B, C, and D are deliberately derived from one paired-risk artifact so their
diffusion noise is identical.  This script fails closed on provenance or
branch-consistency errors before computing any gate metric.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


COMPONENTS = ("collision", "drivable", "progress", "ttc", "comfort", "direction")
SAFETY_NAMES = ("collision", "drivable", "ttc")
LOCKED_GENERATOR_SHA256 = "3a7641d4ac2a9d644eda4cb945d1902d4ac4bebfdad09b156676dc0cbed94e23"
LOCKED_BASE_SHA256 = "59a8de460cfd8b1266c5cdd393372273da5c2465fa6707da551c4ecb1fbd019d"
LOCKED_VERIFIER_SHA256 = "c8f898f21b7e22cf172bdab4b6e4f99bb1537b85c7bf0ed916d23d600ffecbdb"
LOCKED_CALIBRATION_SHA256 = "5d2fc622f4ca7d0ebd9793e2d7f8da466a8e61a89f6d3a0977a15d6396e0e748"
LOCKED_THRESHOLD = 0.7683204412460327
LOCKED_SCHEDULE = {
    "truncation_timestep": 32,
    "roll_timesteps": [32, 24, 16, 8, 0],
    "scheduler_num_inference_steps": 125,
    "scheduler_step_stride": 8,
    "transitions": [[32, 24], [24, 16], [16, 8], [8, 0], [0, -8]],
}
EXPECTED_COUNTS = {
    "fixed-diagnostic": 1024,
    "dev-select": 3072,
    "dev-confirm": 1024,
}
EXPECTED_CACHE_BY_TOKEN_SHA = {
    "fc2f4c607de003688f5fbfc1ab5f97568b3c334e4a34c3cb4ae2fe3de994df25": "training_cache",
    "1d5e2c1ae12f8e5b65edfa9976166b7a24cb77df87fe67b154bff9431a491299": "grpo_dev4096_feature_cache_20260717",
    "c98144d252415f74ccdf041a6f610b9a9bedee6adbba91027a7bb48a2df5554d": "grpo_dev4096_feature_cache_20260717",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def token_set_sha256(tokens):
    return sha256_bytes("\n".join(tokens).encode())


def load_artifact(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("summary"), dict) or not isinstance(payload.get("records"), list):
        raise RuntimeError(f"invalid evaluation artifact schema: {path}")
    return payload


def records_by_token(payload, label):
    result = {}
    for record in payload["records"]:
        token = str(record.get("token"))
        if token in result:
            raise RuntimeError(f"duplicate {label} token: {token}")
        result[token] = record
    return result


def require_close(actual, expected, label, atol=1e-6):
    if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise RuntimeError(f"{label} mismatch: actual={actual!r} expected={expected!r}")


def validate_provenance(original, stage17, expected_count):
    osummary, dsummary = original["summary"], stage17["summary"]
    paired = dsummary.get("paired_risk", {})
    if dsummary.get("checkpoint_sha256") != LOCKED_GENERATOR_SHA256:
        raise RuntimeError("Stage18 generator checkpoint SHA mismatch")
    if dsummary.get("reference_checkpoint_sha256") != LOCKED_BASE_SHA256:
        raise RuntimeError("Stage18 base checkpoint SHA mismatch")
    if paired.get("checkpoint_sha256") != LOCKED_VERIFIER_SHA256:
        raise RuntimeError("Stage18 verifier checkpoint SHA mismatch")
    if dsummary.get("selector_logits_source") != "paired_tail_risk":
        raise RuntimeError("Stage18 selector source is not paired_tail_risk")
    if dsummary.get("generation_policy_algorithm") != "diffgrpo_full_chain":
        raise RuntimeError("Stage18 generation policy algorithm mismatch")
    if dsummary.get("schedule") != LOCKED_SCHEDULE:
        raise RuntimeError("Stage18 full-chain schedule mismatch")
    require_close(float(paired.get("threshold", np.nan)), LOCKED_THRESHOLD, "locked threshold", 1e-12)

    calibration_path = Path(str(paired.get("calibration_path", "")))
    if not calibration_path.is_file():
        raise RuntimeError(f"locked calibration artifact is unavailable: {calibration_path}")
    if sha256_bytes(calibration_path.read_bytes()) != LOCKED_CALIBRATION_SHA256:
        raise RuntimeError("locked calibration artifact SHA mismatch")

    if osummary.get("schedule", {}).get("truncation_timestep") != 8:
        raise RuntimeError("original A is not the locked short schedule")
    if osummary.get("schedule", {}).get("roll_timesteps") != [8, 0]:
        raise RuntimeError("original A roll schedule mismatch")
    if Path(str(osummary.get("checkpoint", ""))).name != "eval_model":
        raise RuntimeError("original A checkpoint is not the frozen base eval_model")
    token_sha = osummary.get("token_set_sha256")
    expected_cache = EXPECTED_CACHE_BY_TOKEN_SHA.get(token_sha)
    if expected_cache is None:
        raise RuntimeError("original A token set is not a locked Stage18 partition")
    if Path(str(dsummary.get("cache_path", ""))).name != expected_cache:
        raise RuntimeError("Stage18 feature cache provenance mismatch")
    for label, summary in (("A", osummary), ("D", dsummary)):
        if summary.get("num_tokens") != expected_count or not summary.get("completed", True):
            raise RuntimeError(f"{label} count/completion provenance mismatch")
        if summary.get("log_split") != "val":
            raise RuntimeError(f"{label} log split is not val")


def validate_and_extract(original, stage17, expected_count):
    validate_provenance(original, stage17, expected_count)
    amap = records_by_token(original, "A")
    dmap = records_by_token(stage17, "D")
    tokens = [str(record["token"]) for record in stage17["records"]]
    if len(tokens) != expected_count or set(tokens) != set(amap):
        raise RuntimeError("Stage18 A/D token pairing or count mismatch")
    for label, payload, ordered_tokens in (
        ("A", original, [str(record["token"]) for record in original["records"]]),
        ("D", stage17, tokens),
    ):
        claimed = payload["summary"].get("token_set_sha256")
        if claimed and claimed != token_set_sha256(ordered_tokens):
            raise RuntimeError(f"{label} token_set_sha256 mismatch")

    av, bv, cv, dv = [], [], [], []
    base_components = {name: [] for name in COMPONENTS}
    selected_components = {name: [] for name in COMPONENTS}
    fallbacks = []
    for token in tokens:
        arecord, drecord = amap[token], dmap[token]
        if drecord.get("selector_logits_source") != "paired_tail_risk":
            raise RuntimeError(f"record selector source mismatch: {token}")
        pair = drecord.get("paired_risk", {})
        fallback = pair.get("fallback")
        if not isinstance(fallback, bool):
            raise RuntimeError(f"invalid fallback flag: {token}")
        require_close(float(pair.get("threshold", np.nan)), LOCKED_THRESHOLD, f"record threshold {token}", 1e-12)
        score = float(pair.get("fallback_score", np.nan))
        if fallback != (score >= LOCKED_THRESHOLD):
            raise RuntimeError(f"fallback decision/score mismatch: {token}")
        current_reward = float(pair["current_reward"])
        base_reward = float(pair["base_reward"])
        require_close(
            float(pair.get("true_current_minus_base", np.nan)),
            current_reward - base_reward,
            f"paired reward delta {token}",
        )
        branch = "base" if fallback else "current"
        branch_reward = base_reward if fallback else current_reward
        require_close(float(drecord["selected_reward"]), branch_reward, f"selected reward branch {token}")

        pair_base_components = np.asarray(pair["base_components"], dtype=np.float64)
        pair_current_components = np.asarray(pair["current_components"], dtype=np.float64)
        if pair_base_components.shape != (len(COMPONENTS),) or pair_current_components.shape != (len(COMPONENTS),):
            raise RuntimeError(f"paired component shape mismatch: {token}")
        branch_components = pair_base_components if fallback else pair_current_components
        for index, name in enumerate(COMPONENTS):
            require_close(
                float(drecord["selected_components"][name]),
                float(branch_components[index]),
                f"selected {name} branch {token}",
            )
            base_components[name].append(float(pair_base_components[index]))
            selected_components[name].append(float(drecord["selected_components"][name]))

        selected_trajectory = np.asarray(drecord["selected_trajectory"], dtype=np.float64)
        branch_trajectory = np.asarray(pair[f"{branch}_trajectory"], dtype=np.float64)
        if selected_trajectory.shape != branch_trajectory.shape or not np.allclose(
            selected_trajectory, branch_trajectory, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"selected trajectory branch mismatch: {token}")

        av.append(float(arecord["selected_reward"]))
        bv.append(base_reward)
        cv.append(current_reward)
        dv.append(float(drecord["selected_reward"]))
        fallbacks.append(fallback)

    arrays = {
        "a": np.asarray(av), "b": np.asarray(bv), "c": np.asarray(cv), "d": np.asarray(dv),
        "fallback": np.asarray(fallbacks, dtype=bool),
        "base_components": {name: np.asarray(values) for name, values in base_components.items()},
        "selected_components": {name: np.asarray(values) for name, values in selected_components.items()},
    }
    for name in ("a", "b", "c", "d"):
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"non-finite reward in {name.upper()}")
    return tokens, arrays


def bootstrap_ci(values, seed=20260721, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or samples % 1000:
        raise ValueError("bootstrap input must be nonempty 1-D and samples divisible by 1000")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples // 1000):
        indices = rng.integers(0, values.size, size=(1000, values.size))
        means.append(values[indices].mean(axis=1))
    return np.quantile(np.concatenate(means), [0.025, 0.975]).tolist()


def evaluate_gate(original, stage17, gate):
    expected = EXPECTED_COUNTS[gate]
    tokens, values = validate_and_extract(original, stage17, expected)
    db = values["d"] - values["b"]
    da = values["d"] - values["a"]
    dc = values["d"] - values["c"]
    cb = values["c"] - values["b"]
    tail_count = max(1, int(np.ceil(0.01 * expected)))
    safety = {
        name: float((values["selected_components"][name] - values["base_components"][name]).mean())
        for name in SAFETY_NAMES
    }
    catastrophic_current = cb <= -0.5
    catastrophic_admitted = catastrophic_current & ~values["fallback"]
    metrics = {
        "a_mean": float(values["a"].mean()),
        "b_mean": float(values["b"].mean()),
        "c_mean": float(values["c"].mean()),
        "d_mean": float(values["d"].mean()),
        "d_minus_b": float(db.mean()),
        "d_minus_b_ci95": bootstrap_ci(db),
        "d_minus_a": float(da.mean()),
        "d_minus_a_ci95": bootstrap_ci(da, 20260722),
        "d_minus_c": float(dc.mean()),
        "d_minus_c_ci95": bootstrap_ci(dc, 20260723),
        "c_minus_b": float(cb.mean()),
        "fallback_count": int(values["fallback"].sum()),
        "fallback_rate": float(values["fallback"].mean()),
        "safety_deltas_d_minus_b": safety,
        "worst_d_minus_b": float(db.min()),
        "bottom_1pct_d_minus_b": float(np.sort(db)[:tail_count].mean()),
        "bottom_1pct_c_minus_b": float(np.sort(cb)[:tail_count].mean()),
        "catastrophic_current_count": int(catastrophic_current.sum()),
        "catastrophic_admitted_count": int(catastrophic_admitted.sum()),
        "catastrophic_admitted_rate": float(catastrophic_admitted.mean()),
    }
    failures = []
    if gate == "dev-select":
        if metrics["d_minus_b"] < 0.015 or metrics["d_minus_b_ci95"][0] <= 0:
            failures.append("D-B mean/CI gate failed")
        if metrics["d_minus_a"] < 0.020 or metrics["d_minus_a_ci95"][0] <= 0:
            failures.append("D-A mean/CI gate failed")
        if metrics["d_minus_c"] < 0:
            failures.append("D-C mean gate failed")
        if any(delta < 0 for delta in safety.values()):
            failures.append("safety component declined relative to B")
        if metrics["fallback_rate"] > 0.35:
            failures.append("fallback rate exceeds 35%")
    elif gate == "dev-confirm":
        if metrics["d_minus_b"] < 0.008 or metrics["d_minus_b_ci95"][0] <= 0:
            failures.append("D-B mean/CI gate failed")
        if metrics["d_minus_a"] < 0.012 or metrics["d_minus_a_ci95"][0] <= 0:
            failures.append("D-A mean/CI gate failed")
        if metrics["d_minus_c"] < 0:
            failures.append("D-C mean gate failed")
        if any(delta < 0 for delta in safety.values()):
            failures.append("safety component declined relative to B")
    result = {
        "passed": not failures,
        "decision_scope": "diagnostic_only" if gate == "fixed-diagnostic" else "average_primary_gate",
        "gate": gate,
        "expected_tokens": expected,
        "token_set_sha256": token_set_sha256(tokens),
        "paired_noise_definition": {
            "B": "stage17.records[].paired_risk.base_*",
            "C": "stage17.records[].paired_risk.current_*",
            "D": "stage17.records[].selected_*",
        },
        "locked_provenance": {
            "generator_sha256": LOCKED_GENERATOR_SHA256,
            "base_sha256": LOCKED_BASE_SHA256,
            "verifier_sha256": LOCKED_VERIFIER_SHA256,
            "calibration_sha256": LOCKED_CALIBRATION_SHA256,
            "threshold": LOCKED_THRESHOLD,
            "schedule": LOCKED_SCHEDULE,
        },
        "tail_metrics_are_nonblocking": True,
        "metrics": metrics,
        "failures": failures,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=tuple(EXPECTED_COUNTS), required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--stage17", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_gate(load_artifact(args.original), load_artifact(args.stage17), args.gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
