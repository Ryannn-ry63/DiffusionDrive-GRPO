#!/usr/bin/env python3
"""Apply locked Stage-17 fixed1024 or dev-select3072 gates."""

import argparse
import json
from pathlib import Path

import numpy as np


SAFETY_NAMES = ("collision", "drivable", "ttc")


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, {r["token"]: r for r in payload["records"]}


def bootstrap(values, seed=20260721, samples=10000):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples // 1000):
        idx = rng.integers(0, values.size, size=(1000, values.size))
        means.extend(values[idx].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("fixed1024", "dev-select"), required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--base-full", type=Path, required=True)
    parser.add_argument("--stage16", type=Path, required=True)
    parser.add_argument("--stage17", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pa, a = load(args.original); pb, b = load(args.base_full)
    pc, c = load(args.stage16); pd, d = load(args.stage17)
    tokens = [r["token"] for r in pd["records"]]
    expected = 1024 if args.gate == "fixed1024" else 3072
    if len(tokens) != expected or any(set(x) != set(tokens) for x in (a, b, c, d)):
        raise RuntimeError("Stage-17 gate token pairing/count mismatch")
    av = np.asarray([a[t]["selected_reward"] for t in tokens])
    bv = np.asarray([b[t]["selected_reward"] for t in tokens])
    cv = np.asarray([c[t]["selected_reward"] for t in tokens])
    dv = np.asarray([d[t]["selected_reward"] for t in tokens])
    db, da, dc, cb = dv-bv, dv-av, dv-cv, cv-bv
    tail_count = max(1, int(np.ceil(0.01 * expected)))
    component_delta = {}
    for name in SAFETY_NAMES:
        component_delta[name] = float(np.mean([
            d[t]["selected_components"][name] - b[t]["selected_components"][name]
            for t in tokens
        ]))
    fallback_rate = float(pd["summary"]["paired_risk"]["fallback_rate"])
    metrics = {
        "d_minus_b": float(db.mean()), "d_minus_b_ci95": bootstrap(db),
        "d_minus_a": float(da.mean()), "d_minus_a_ci95": bootstrap(da, 20260722),
        "d_minus_c": float(dc.mean()), "d_minus_c_ci95": bootstrap(dc, 20260723),
        "c_minus_b": float(cb.mean()),
        "worst_d_minus_b": float(db.min()),
        "bottom_1pct_d_minus_b_cvar": float(np.sort(db)[:tail_count].mean()),
        "bottom_1pct_c_minus_b_cvar": float(np.sort(cb)[:tail_count].mean()),
        "fallback_rate": fallback_rate,
        "safety_deltas": component_delta,
    }
    failures = []
    if metrics["d_minus_b"] < 0.010 or metrics["d_minus_b_ci95"][0] <= 0:
        failures.append("D-B mean/CI gate failed")
    if metrics["d_minus_a"] < 0.015 or metrics["d_minus_a_ci95"][0] <= 0:
        failures.append("D-A mean/CI gate failed")
    required_dc = 0.0 if args.gate == "fixed1024" else 0.001
    if metrics["d_minus_c"] < required_dc:
        failures.append("D-C mean gate failed")
    if args.gate == "dev-select" and metrics["d_minus_c_ci95"][0] < 0:
        failures.append("D-C CI gate failed")
    if any(value < 0 for value in component_delta.values()):
        failures.append("safety component declined")
    if metrics["worst_d_minus_b"] <= -0.5:
        failures.append("worst D-B token gate failed")
    if metrics["bottom_1pct_d_minus_b_cvar"] < -0.1:
        failures.append("D-B bottom-1% CVaR gate failed")
    if metrics["bottom_1pct_d_minus_b_cvar"] <= metrics["bottom_1pct_c_minus_b_cvar"]:
        failures.append("Stage-17 did not improve Stage-16 tail CVaR")
    if fallback_rate > 0.35:
        failures.append("fallback rate exceeds 35%")
    result = {"passed": not failures, "gate": args.gate, "expected_tokens": expected, "metrics": metrics, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
