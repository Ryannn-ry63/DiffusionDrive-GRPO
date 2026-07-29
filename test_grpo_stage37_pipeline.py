import json
from pathlib import Path

import numpy as np

from scripts.evaluation.calibrate_grpo_stage37_jfi import (
    select_modes,
    upper95,
)


ROOT = Path(__file__).resolve().parent
PLAN_SHA = "4dc469dd0be4d2579796ff70ae7a09ed9fd8e4c92449cd057e42463d268bb5d8"


def test_folds23_fit_manifest_excludes_calibration_and_test_folds():
    freeze = json.loads((
        ROOT / "artifacts/grpo_stage37/jfi/manifests/freeze.json"
    ).read_text())
    manifest = json.loads(Path(freeze["train_manifest"]).read_text())
    assert freeze["plan_sha256"] == PLAN_SHA
    assert freeze["count"] == 2035
    assert freeze["num_logs"] == 303
    assert manifest["summary"]["source_folds"] == [2, 3]
    assert manifest["summary"]["excluded_folds"] == [0, 1]
    assert {record["stage37_jfi_source_fold"] for record in manifest["records"]} == {2, 3}


def test_numpy_calibration_selection_matches_four_candidate_cap():
    eligible = np.ones((2, 20), dtype=bool)
    q10 = np.tile(np.arange(20, dtype=np.float64), (2, 1))
    fallback = np.asarray([19, 0])
    selected, pool = select_modes(eligible, q10, fallback)
    assert selected.tolist() == [18, 19]
    assert pool.tolist() == [4, 4]


def test_zero_catastrophes_need_enough_calibration_scenes():
    assert upper95(0, 100) > 0.005
    assert upper95(0, 1000) < 0.005


def test_pipeline_scripts_freeze_generator_attribution_and_stop_before_navtest():
    generator = (ROOT / "scripts/evaluation/summarize_grpo_stage37_generator.py").read_text()
    factorial = (ROOT / "scripts/evaluation/summarize_grpo_stage37_factorial.py").read_text()
    plan = (ROOT / "GRPO_STAGE37_BISTATE_PROJECTED_JFI_PLAN_20260728.md").read_text()
    assert "selected_gain_at_least_0.002" in generator
    assert "generator_same_jfi_at_least_0.001" in factorial
    assert "gain_at_least_0.005" in factorial
    assert "No folds 2/3 expansion and no navtest evaluation" in plan
