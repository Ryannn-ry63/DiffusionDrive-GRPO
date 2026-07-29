#!/usr/bin/env python3
"""Summarize the frozen Stage37 2x2 and enforce the full-system gate."""

from __future__ import annotations

import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np

PUBLIC_SHA="008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
PLAN_SHA="4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"
NOISES=(20261511,20261512); COMPONENTS=("collision","drivable","ttc")


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()


def load(path:Path,checkpoint_sha:str,domain:str,noise:int,selector:str,jfi_sha:str="",calibration_sha:str="")->dict:
 p=json.loads(path.read_text());s=p["summary"];records=p["records"]
 if not s.get("completed") or s.get("checkpoint_sha256")!=checkpoint_sha or s.get("generator_domain")!=domain or s.get("evaluation_noise_namespace")!=noise:raise RuntimeError(f"factorial provenance drift: {path}")
 if selector=="stage25":
  if s.get("selector_logits_source")!="trajectory_relative_harm_v3":raise RuntimeError("Stage25 source drift")
 else:
  j=s["stage37_jfi_selector"]
  if s.get("selector_logits_source")!="joint_feasible_improvement_v1" or j.get("checkpoint_sha256")!=jfi_sha or j.get("calibration_sha256")!=calibration_sha:raise RuntimeError("JFI source/SHA drift")
 out={"tokens":[],"logs":[],"selected":[],"pool":[],"components":{x:[] for x in COMPONENTS}}
 for r in records:
  out["tokens"].append(r["token"]);out["logs"].append(r["log_name"]);out["selected"].append(float(r["selected_reward"]))
  for name in COMPONENTS:out["components"][name].append(float(r["selected_components"][name]))
  if selector=="jfi":out["pool"].append(float(r["stage37_jfi_selector"]["candidate_pool_size"]))
 out["selected"]=np.asarray(out["selected"]);out["pool"]=np.asarray(out["pool"])
 out["components"]={k:np.asarray(v) for k,v in out["components"].items()};return out


def bootstrap(values:np.ndarray,logs:list[str])->list[float]:
 g=defaultdict(list)
 for v,n in zip(values,logs):g[n].append(float(v))
 names=sorted(g);s=np.asarray([sum(g[x]) for x in names]);c=np.asarray([len(g[x]) for x in names]);rng=np.random.default_rng(20373737);samples=np.empty(10000)
 for start in range(0,10000,500):
  idx=rng.integers(0,len(names),(500,len(names)));samples[start:start+500]=s[idx].sum(1)/c[idx].sum(1)
 return np.quantile(samples,(0.025,0.975)).tolist()


def trimmed(values:np.ndarray)->float:
 n=int(np.floor(len(values)*.1));x=np.sort(values);return float((x[n:-n] if n else x).mean())


def main()->None:
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--baseline-root",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 selection=json.loads((a.root/"pilot/generator_selection.json").read_text());jfi_freeze=json.loads((a.root/"jfi/training/formal/checkpoints.json").read_text());jfi_entry=jfi_freeze["checkpoints"][0];jfi_sha=jfi_entry["sha256"]
 if not selection.get("passed") or selection.get("plan_sha256")!=PLAN_SHA or not jfi_freeze.get("passed"):raise RuntimeError("Stage37 selection/JFI freeze drift")
 step=int(selection["selected_step"]);vectors=defaultdict(list);component_vectors={x:[] for x in COMPONENTS};folds=[];noises=[];logs=[];pools=[];calibrated_pools=[];absolute=defaultdict(list);artifact_shas={}
 for fold in (0,1):
  gen=selection["folds"][str(fold)];cal_path=a.root/f"jfi/calibration/h{fold}/calibration.json";cal=json.loads(cal_path.read_text());cal_sha=sha(cal_path)
  if not cal.get("passed") or cal.get("selector_checkpoint_sha256")!=jfi_sha:raise RuntimeError("JFI calibration drift")
  calibrated_pools.append(float(cal["mean_candidate_pool_size"]))
  for noise in NOISES:
   paths={
    "P_S25":a.baseline_root/f"fold{fold}/P_ns{noise}.json",
    "BPD_S25":a.root/f"pilot/generator_eval/fold{fold}/BPD{step}_ns{noise}.json",
    "P_JFI":a.root/f"pilot/factorial/fold{fold}/P_JFI_ns{noise}.json",
    "BPD_JFI":a.root/f"pilot/factorial/fold{fold}/BPD_JFI_ns{noise}.json"}
   for path in paths.values():artifact_shas[str(path.resolve())]=sha(path)
   systems={
    "P_S25":load(paths["P_S25"],PUBLIC_SHA,f"stage34_pilot_p_fold{fold}",noise,"stage25"),
    "BPD_S25":load(paths["BPD_S25"],gen["checkpoint_sha256"],f"stage37_pilot_bpd{step}_fold{fold}",noise,"stage25"),
    "P_JFI":load(paths["P_JFI"],PUBLIC_SHA,f"stage37_factorial_p_jfi_fold{fold}",noise,"jfi",jfi_sha,cal_sha),
    "BPD_JFI":load(paths["BPD_JFI"],gen["checkpoint_sha256"],f"stage37_factorial_bpd_jfi_fold{fold}",noise,"jfi",jfi_sha,cal_sha)}
   if len({tuple(x["tokens"]) for x in systems.values()})!=1:raise RuntimeError("2x2 token order drift")
   full=systems["BPD_JFI"]["selected"]-systems["P_S25"]["selected"];gen_effect=systems["BPD_JFI"]["selected"]-systems["P_JFI"]["selected"]
   vectors["full"].append(full);vectors["generator_same_jfi"].append(gen_effect);vectors["selector_public"].append(systems["P_JFI"]["selected"]-systems["P_S25"]["selected"]);vectors["generator_stage25"].append(systems["BPD_S25"]["selected"]-systems["P_S25"]["selected"])
   for name in COMPONENTS:component_vectors[name].append(systems["BPD_JFI"]["components"][name]-systems["P_S25"]["components"][name])
   count=len(full);folds.append(np.full(count,fold));noises.append(np.full(count,noise));logs.extend(f"{fold}:{noise}:{x}" for x in systems["P_S25"]["logs"]);pools.append(systems["BPD_JFI"]["pool"])
   for name,x in systems.items():absolute[name].append(x["selected"])
 v={k:np.concatenate(x) for k,x in vectors.items()};fold_ids=np.concatenate(folds);noise_ids=np.concatenate(noises);pool=np.concatenate(pools);components={k:np.concatenate(x) for k,x in component_vectors.items()};wins=int(np.count_nonzero(v["full"]>0));losses=int(np.count_nonzero(v["full"]<0))
 metrics={"mean_full_gain":float(v["full"].mean()),"mean_generator_effect_same_jfi":float(v["generator_same_jfi"].mean()),"mean_public_jfi_selector_effect":float(v["selector_public"].mean()),"mean_generator_effect_stage25":float(v["generator_stage25"].mean()),"whole_log_bootstrap_ci95":bootstrap(v["full"],logs),"fold_means":{str(f):float(v["full"][fold_ids==f].mean()) for f in (0,1)},"namespace_means":{str(n):float(v["full"][noise_ids==n].mean()) for n in NOISES},"component_mean_deltas":{k:float(x.mean()) for k,x in components.items()},"wins":wins,"losses":losses,"ties":int(len(v["full"])-wins-losses),"trimmed10_mean":trimmed(v["full"]),"catastrophic_count":int(np.count_nonzero(v["full"]<=-.5)),"test_mean_candidate_pool_size":float(pool.mean()),"calibrated_mean_candidate_pool_size":float(np.mean(calibrated_pools)),"max_candidate_pool_size":int(pool.max()),"absolute_means":{k:float(np.concatenate(x).mean()) for k,x in absolute.items()},"stretch_target_gain":.009,"stretch_target_met":float(v["full"].mean())>=.009}
 checks={"gain_at_least_0.005":metrics["mean_full_gain"]>=.005,"generator_same_jfi_at_least_0.001":metrics["mean_generator_effect_same_jfi"]>=.001,"whole_log_ci_lower_positive":metrics["whole_log_bootstrap_ci95"][0]>0,"fold_means_positive":all(x>0 for x in metrics["fold_means"].values()),"namespace_means_positive":all(x>0 for x in metrics["namespace_means"].values()),"guard_components_no_worse_0.0005":all(x>=-.0005 for x in metrics["component_mean_deltas"].values()),"wins_exceed_losses":wins>losses,"trimmed10_nonnegative":metrics["trimmed10_mean"]>=0,"catastrophic_count_zero":metrics["catastrophic_count"]==0,"calibrated_mean_pool_between_2_and_4":2<=metrics["calibrated_mean_candidate_pool_size"]<=4,"max_pool_at_most_4":metrics["max_candidate_pool_size"]<=4};checks["passed"]=all(checks.values())
 result={"schema_version":1,"stage":37,"plan_sha256":PLAN_SHA,"pilot_oof":True,"factorial":["Public+Stage25","Stage37+Stage25","Public+JFI","Stage37+JFI"],"selected_step":step,"generator_gate_passed":True,"metrics":metrics,"promotion_checks":checks,"pilot_passed":checks["passed"],"jfi_selector_checkpoint_sha256":jfi_sha,"input_artifact_shas":artifact_shas}
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
