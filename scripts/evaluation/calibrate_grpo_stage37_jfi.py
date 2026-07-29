#!/usr/bin/env python3
"""Safety-first cross-fitted calibration for the Stage37 JFI selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


PLAN_SHA = "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
GUARDS = (0, 1, 3)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upper95(events: int, trials: int) -> float:
    return 1.0 if events == trials else float(beta.ppf(0.95, events + 1, trials - events))


def select_modes(eligible: np.ndarray, q10: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eligible = eligible.copy(); eligible[np.arange(len(fallback)), fallback] = False
    scores = np.where(eligible, q10, -np.inf)
    switched = eligible.any(axis=1)
    challenger = scores.argmax(axis=1)
    pool = np.minimum(eligible.sum(axis=1), 3) + 1
    return np.where(switched, challenger, fallback), pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, choices=(0, 1), required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--base-selector-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selector_sha = sha(args.selector_checkpoint)
    base = json.loads(args.base_selector_calibration.read_text())
    if not base.get("passed") or "ood" not in base or float(base.get("ood_threshold", -1)) < 0:
        raise RuntimeError("frozen Stage25 OOD calibration drifted")
    ood_mean = np.asarray(base["ood"]["mean"], dtype=np.float64)
    ood_var = np.asarray(base["ood"]["variance"], dtype=np.float64)
    ood_threshold = float(base["ood_threshold"])

    rows = []; provenance=[]; seen=set()
    for path in args.artifact:
        payload=json.loads(path.read_text()); summary=payload["summary"]; jfi=summary["stage37_jfi_selector"]
        if jfi.get("checkpoint_sha256") != selector_sha or not jfi.get("calibration_collection"):
            raise RuntimeError(f"invalid JFI calibration provenance: {path}")
        domain=str(summary.get("generator_domain","")); noise=int(summary.get("evaluation_noise_namespace",-1))
        for record in payload["records"]:
            key=(domain,noise,record["token"])
            if key in seen: raise RuntimeError("duplicate JFI calibration scene")
            seen.add(key); diagnostic=record["stage37_jfi_selector"]
            reward=np.asarray(record["candidate_rewards"],dtype=np.float64); component=np.asarray(record["candidate_components"],dtype=np.float64)
            joint=np.asarray(diagnostic["joint_probabilities"],dtype=np.float64); quantile=np.asarray(diagnostic["delta_quantiles"],dtype=np.float64)
            embedding=np.asarray(diagnostic["embedding_mean"],dtype=np.float64); fallback=int(diagnostic["fallback_mode"])
            if reward.shape!=(20,) or component.shape!=(20,6) or joint.shape!=(20,8) or quantile.shape!=(20,8,3) or embedding.shape!=(20,ood_mean.size):
                raise RuntimeError("JFI calibration shape drifted")
            if not all(np.isfinite(x).all() for x in (reward,component,joint,quantile,embedding)):
                raise RuntimeError("JFI calibration contains non-finite values")
            rows.append((domain,noise,record["token"],record["log_name"],fallback,reward,component,joint,quantile,embedding))
        provenance.append({"path":str(path),"sha256":sha(path),"domain":domain,"namespace":noise})
    if not rows: raise RuntimeError("JFI calibration is empty")
    fallback=np.asarray([x[4] for x in rows]); reward=np.stack([x[5] for x in rows]); component=np.stack([x[6] for x in rows])
    joint=np.stack([x[7] for x in rows]); quantile=np.stack([x[8] for x in rows]); embedding=np.stack([x[9] for x in rows])
    joint_lcb=joint.mean(2)-1.96*joint.std(2); q10=quantile[:,:,:,0].mean(2)
    ood=((embedding-ood_mean)**2/ood_var).mean(2); index=np.arange(len(rows)); fallback_reward=reward[index,fallback]; fallback_component=component[index,fallback]
    joint_grid=np.unique(np.concatenate((np.linspace(0,1,101),np.quantile(np.clip(joint_lcb,0,1),np.linspace(0,1,101)))))
    q_values=q10[np.isfinite(q10)]; q_grid=np.unique(np.concatenate(([q_values.min()-1e-9,0.0],np.quantile(q_values,np.linspace(0,0.95,40)))))
    best=None; feasible=[]
    for threshold in joint_grid:
        joint_ok=(joint_lcb>=threshold)&(ood<=ood_threshold)
        for floor in q_grid:
            selected,pool=select_modes(joint_ok&(q10>=floor),q10,fallback)
            delta=reward[index,selected]-fallback_reward; component_delta=component[index,selected]-fallback_component
            catastrophic=int(np.count_nonzero(delta<=-0.5)); upper=upper95(catastrophic,len(delta)); guard=component_delta[:,GUARDS].mean(0); mean_pool=float(pool.mean())
            if upper>0.005 or np.any(guard < -0.0005) or not 2.0<=mean_pool<=4.0 or int(pool.max())>4: continue
            metrics={"joint_threshold":float(threshold),"q10_floor":float(floor),"mean_candidate_pool_size":mean_pool,"max_candidate_pool_size":int(pool.max()),"switch_rate":float(np.mean(selected!=fallback)),"selected_minus_fallback_mean":float(delta.mean()),"catastrophic_count":catastrophic,"catastrophic_rate_upper95":upper,"guard_component_deltas":{"collision":float(guard[0]),"drivable":float(guard[1]),"ttc":float(guard[2])},"wins":int(np.count_nonzero(delta>0)),"losses":int(np.count_nonzero(delta<0))}
            # Safety is lexicographically primary; gain is considered only after it.
            rank=(-upper,float(guard.min()),float(delta.mean()),-abs(mean_pool-3.0),float(threshold),float(floor))
            feasible.append(metrics)
            if best is None or rank>best[0]: best=(rank,metrics)
    result={"schema_version":1,"stage":37,"holdout_fold":args.holdout,"calibration_fold":1-args.holdout,"passed":best is not None,"plan_sha256":PLAN_SHA,"selector_checkpoint":str(args.selector_checkpoint),"selector_checkpoint_sha256":selector_sha,"base_selector_calibration":str(args.base_selector_calibration),"base_selector_calibration_sha256":sha(args.base_selector_calibration),"num_scenes":len(rows),"num_artifacts":len(args.artifact),"confidence_z":1.96,"ood_threshold":ood_threshold,"ood":base["ood"],"safety_constraints":{"catastrophic_upper95_max":0.005,"component_delta_min":-0.0005,"mean_pool_min":2.0,"mean_pool_max":4.0,"max_candidates":4},"num_feasible_calibrations":len(feasible),"artifact_provenance":provenance}
    if best is not None: result.update(best[1])
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if best is None: raise RuntimeError("no JFI calibration satisfies the frozen safety/pool gates")


if __name__ == "__main__": main()
