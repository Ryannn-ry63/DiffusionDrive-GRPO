#!/usr/bin/env python3
"""Apply the preregistered Stage37 generator-only gate and freeze one step."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PUBLIC_SHA = "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
SELECTOR_SHA = "023d7b6b77bb8fa2dc3e779849f3716688abf0796b019416494c80b29615b691"
CALIBRATION_SHA = "fec2a10573e913e4452f8eea92e6334fcd5e6f1104e6244876770973a1f41990"
PLAN_SHA = "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
BUCKET_SHA = "e067680e9d601929b69b932c59de1f092e3c229d6aeb0bf5ee5aa0408b9a7cf7"
STEPS = (48, 96, 144, 192)
NOISES = (20261511, 20261512)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_entry(root: Path, fold: int, step: int) -> dict:
    path = root / f"generator/fold{fold}/formal/checkpoints.json"
    freeze = json.loads(path.read_text())
    matches = [x for x in freeze["checkpoints"] if int(x["global_step"]) == step]
    if (
        not freeze.get("passed") or freeze.get("stage") != 37
        or freeze.get("plan_sha256") != PLAN_SHA or len(matches) != 1
    ):
        raise RuntimeError("Stage37 generator freeze drifted")
    return matches[0]


def historical_rgt_entry(root: Path, fold: int) -> dict:
    path = root / f"pilot/training/RGT/fold{fold}/formal/checkpoints.json"
    freeze = json.loads(path.read_text())
    matches = [x for x in freeze["checkpoints"] if int(x["global_step"]) == 192]
    if not freeze.get("passed") or freeze.get("stage") != 36 or len(matches) != 1:
        raise RuntimeError("Stage36 RGT192 freeze drifted")
    return matches[0]


def load(path: Path, checkpoint_sha: str, domain: str, noise: int) -> dict:
    payload = json.loads(path.read_text()); summary = payload["summary"]
    records = payload["records"]; selector = summary["stage25_selector"]
    if not all((
        summary.get("completed") is True,
        summary.get("num_failures") == 0,
        summary.get("checkpoint_sha256") == checkpoint_sha,
        summary.get("reference_checkpoint_sha256") == PUBLIC_SHA,
        summary.get("generator_domain") == domain,
        summary.get("evaluation_noise_namespace") == noise,
        summary.get("selector_logits_source") == "trajectory_relative_harm_v3",
        selector.get("checkpoint_sha256") == SELECTOR_SHA,
        selector.get("calibration_sha256") == CALIBRATION_SHA,
    )):
        raise RuntimeError(f"Stage37 generator artifact provenance drifted: {path}")
    output = defaultdict(list)
    for record in records:
        rewards = np.asarray(record["candidate_rewards"], dtype=np.float64)
        eligible = np.asarray(record["stage24_selector"]["eligible"], dtype=bool)
        fallback = int(record["stage24_selector"]["fallback_mode"])
        if rewards.shape != (20,) or eligible.shape != (20,) or not np.isfinite(rewards).all():
            raise RuntimeError(f"invalid all-20 record: {path}")
        eligible[fallback] = True
        output["tokens"].append(record["token"]); output["logs"].append(record["log_name"])
        output["selected"].append(float(record["selected_reward"]))
        output["candidate_mean"].append(float(rewards.mean()))
        output["candidate_rewards"].append(rewards)
        output["raw_oracle"].append(float(rewards.max()))
        output["safe_oracle"].append(float(rewards[eligible].max()))
    return {
        "tokens": output["tokens"], "logs": output["logs"],
        **{name: np.asarray(output[name]) for name in ("selected", "candidate_mean", "raw_oracle", "safe_oracle")},
        "candidate_rewards": np.stack(output["candidate_rewards"]),
    }


def bootstrap(values: np.ndarray, logs: list[str], seed: int) -> list[float]:
    groups = defaultdict(list)
    for value, log_name in zip(values, logs): groups[log_name].append(float(value))
    names = sorted(groups); sums = np.asarray([sum(groups[x]) for x in names]); counts = np.asarray([len(groups[x]) for x in names])
    rng = np.random.default_rng(seed); samples = np.empty(10000)
    for start in range(0, 10000, 500):
        index = rng.integers(0, len(names), (500, len(names)))
        samples[start:start+500] = sums[index].sum(1) / counts[index].sum(1)
    return np.quantile(samples, (0.025, 0.975)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage37-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--stage36-root", type=Path, required=True)
    parser.add_argument("--bucket-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.bucket_manifest) != BUCKET_SHA: raise RuntimeError("bucket manifest drifted")
    bucket_payload = json.loads(args.bucket_manifest.read_text())
    bucket = {token:int(record["bucket_id"]) for token,record in bucket_payload["tokens"].items()}
    comparisons = {}; artifact_shas = {}; entries = {}
    for step in STEPS:
        vectors = defaultdict(list); fold_ids=[]; noise_ids=[]; bucket_ids=[]; logs=[]
        for fold in (0,1):
            current_entry = checkpoint_entry(args.stage37_root, fold, step); entries[(fold,step)] = current_entry
            rgt_entry = historical_rgt_entry(args.stage36_root, fold)
            for noise in NOISES:
                public_path=args.baseline_root/f"fold{fold}/P_ns{noise}.json"
                current_path=args.stage37_root/f"pilot/generator_eval/fold{fold}/BPD{step}_ns{noise}.json"
                rgt_path=args.stage36_root/f"pilot/eval/fold{fold}/RGT192_ns{noise}.json"
                for path in (public_path,current_path,rgt_path): artifact_shas[str(path.resolve())]=sha(path)
                public=load(public_path,PUBLIC_SHA,f"stage34_pilot_p_fold{fold}",noise)
                current=load(current_path,current_entry["sha256"],f"stage37_pilot_bpd{step}_fold{fold}",noise)
                rgt=load(rgt_path,rgt_entry["sha256"],f"stage36_pilot_rgt192_fold{fold}",noise)
                if not public["tokens"]==current["tokens"]==rgt["tokens"]: raise RuntimeError("paired token order drifted")
                top5=np.argpartition(public["candidate_rewards"],-5,axis=1)[:,-5:]; row=np.arange(len(top5))[:,None]
                for name in ("selected","candidate_mean","raw_oracle","safe_oracle"):
                    vectors[name].append(current[name]-public[name])
                vectors["top5"].append((current["candidate_rewards"][row,top5]-public["candidate_rewards"][row,top5]).mean(1))
                vectors["vs_rgt"].append(current["selected"]-rgt["selected"])
                count=len(public["tokens"]); fold_ids.append(np.full(count,fold)); noise_ids.append(np.full(count,noise))
                bucket_ids.append(np.asarray([bucket[x] for x in public["tokens"]])); logs.extend(f"{fold}:{noise}:{x}" for x in public["logs"])
        v={name:np.concatenate(parts) for name,parts in vectors.items()}; folds=np.concatenate(fold_ids); noises=np.concatenate(noise_ids); buckets=np.concatenate(bucket_ids)
        summary={
            "count":int(v["selected"].size), "mean_selected_delta":float(v["selected"].mean()),
            "mean_candidate_delta":float(v["candidate_mean"].mean()), "mean_raw_oracle_delta":float(v["raw_oracle"].mean()),
            "mean_safe_oracle_delta":float(v["safe_oracle"].mean()), "mean_public_top5_delta":float(v["top5"].mean()),
            "mean_selected_delta_vs_stage36_rgt192":float(v["vs_rgt"].mean()),
            "whole_log_bootstrap_ci95":bootstrap(v["selected"],logs,203700+step),
            "fold_means":{str(f):float(v["selected"][folds==f].mean()) for f in (0,1)},
            "namespace_means":{str(n):float(v["selected"][noises==n].mean()) for n in NOISES},
            "hard_scene_gain":float(v["selected"][buckets<3].mean()), "mature_scene_gain":float(v["selected"][buckets==3].mean()),
        }
        checks={
            "selected_gain_at_least_0.002":summary["mean_selected_delta"]>=0.002,
            "whole_log_ci_lower_positive":summary["whole_log_bootstrap_ci95"][0]>0,
            "fold_means_positive":all(x>0 for x in summary["fold_means"].values()),
            "namespace_means_positive":all(x>0 for x in summary["namespace_means"].values()),
            "public_top5_nonnegative":summary["mean_public_top5_delta"]>=0,
            "candidate_mean_nonnegative":summary["mean_candidate_delta"]>=0,
            "raw_oracle_nonnegative":summary["mean_raw_oracle_delta"]>=0,
            "safe_oracle_nonnegative":summary["mean_safe_oracle_delta"]>=0,
            "hard_gain_at_least_0.008":summary["hard_scene_gain"]>=0.008,
            "mature_gain_at_least_negative_0.0001":summary["mature_scene_gain"]>=-0.0001,
            "no_worse_than_stage36_rgt192":summary["mean_selected_delta_vs_stage36_rgt192"]>=0,
        }; checks["passed"]=all(checks.values()); summary["promotion_checks"]=checks; comparisons[str(step)]=summary
    passing=[step for step in STEPS if comparisons[str(step)]["promotion_checks"]["passed"]]
    selected=max(passing,key=lambda step:comparisons[str(step)]["mean_selected_delta"]) if passing else None
    result={
        "schema_version":1,"stage":37,"pilot_oof":True,"plan_sha256":PLAN_SHA,"objective_revision":"bistate_projected_deployment_v1",
        "comparisons":comparisons,"passing_steps":passing,"pilot_passed":bool(passing),"selected_step":selected,
        "stop_for_negative_public_top5_all_steps":all(comparisons[str(x)]["mean_public_top5_delta"]<0 for x in STEPS),
        "input_artifact_shas":artifact_shas,
    }
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")
    if selected is not None:
        selection={"schema_version":1,"stage":37,"passed":True,"plan_sha256":PLAN_SHA,"selected_step":selected,"generator_gate":comparisons[str(selected)],"folds":{}}
        for fold in (0,1):
            item=entries[(fold,selected)]; selection["folds"][str(fold)]={"checkpoint":item["path"],"checkpoint_sha256":item["sha256"]}
        if args.selection_output.exists(): raise FileExistsError(args.selection_output)
        args.selection_output.parent.mkdir(parents=True,exist_ok=True); args.selection_output.write_text(json.dumps(selection,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
