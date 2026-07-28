from pathlib import Path

import numpy as np

from scripts.evaluation.summarize_grpo_stage35_pilot import (
    expected_checkpoint,
    summarize,
    trimmed10_mean,
)


BASELINE_EVAL_ROOT = Path("artifacts/grpo_stage34/pilot/eval")


def test_stage35_reuses_exact_paired_public_and_dpel_controls():
    public_sha, public_domain = expected_checkpoint(
        BASELINE_EVAL_ROOT, "P", 0
    )
    dpel_sha, dpel_domain = expected_checkpoint(
        BASELINE_EVAL_ROOT, "DPEL192", 1
    )
    assert len(public_sha) == len(dpel_sha) == 64
    assert public_domain == "stage34_pilot_p_fold0"
    assert dpel_domain == "stage34_pilot_dpel192_fold1"


def test_stage35_summary_preserves_fold_namespace_and_tail_guards():
    count = 40
    selected = np.full(count, 0.002)
    buckets = np.tile(np.arange(4), 10)
    folds = np.repeat((0, 1), count // 2)
    noises = np.tile(np.repeat((20261511, 20261512), 10), 2)
    result = summarize(
        selected_delta=selected,
        candidate_delta=np.full(count, 0.001),
        raw_oracle_delta=np.full(count, 0.0015),
        safe_oracle_delta=np.full(count, 0.001),
        top5_delta=np.full(count, 0.0005),
        component_deltas={
            name: np.zeros(count)
            for name in ("collision", "drivable", "ttc")
        },
        buckets=buckets,
        logs=[
            f"{folds[index]}:{noises[index]}:log{index // 2}"
            for index in range(count)
        ],
        folds=folds,
        noises=noises,
        seed=35,
    )
    assert result["mean_selected_delta"] == 0.002
    assert result["mean_safe_deployable_oracle_delta"] == 0.001
    assert set(result["fold_mean_selected_delta"]) == {"0", "1"}
    assert set(result["namespace_mean_selected_delta"]) == {
        "20261511",
        "20261512",
    }
    assert result["whole_log_bootstrap_ci95"][0] > 0
    assert result["wins"] == count
    assert result["catastrophic_count_delta_at_most_-0.5"] == 0


def test_stage35_trimmed_mean_discards_symmetric_tails():
    values = np.concatenate(([-100.0], np.ones(8), [100.0]))
    assert trimmed10_mean(values) == 1.0
