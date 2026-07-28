from pathlib import Path

import numpy as np

from scripts.evaluation.summarize_grpo_stage34_pilot import (
    expected_checkpoint,
    summarize,
    trimmed10_mean,
)


EVAL_ROOT = Path("artifacts/grpo_stage34/pilot/eval")


def test_stage34_dpel_positive_control_resolves_from_frozen_checkpoint():
    checkpoint_sha, domain = expected_checkpoint(EVAL_ROOT, "DPEL192", 0)
    assert len(checkpoint_sha) == 64
    assert domain == "stage34_pilot_dpel192_fold0"


def test_stage34_summary_keeps_fold_namespace_and_ceiling_statistics():
    count = 40
    selected = np.full(count, 0.002)
    candidate = np.full(count, 0.001)
    raw_oracle = np.full(count, 0.0015)
    safe_oracle = np.full(count, 0.001)
    top5 = np.full(count, 0.0005)
    buckets = np.tile(np.arange(4), 10)
    folds = np.repeat((0, 1), count // 2)
    noises = np.tile(np.repeat((20261511, 20261512), 10), 2)
    result = summarize(
        selected_delta=selected,
        candidate_delta=candidate,
        raw_oracle_delta=raw_oracle,
        safe_oracle_delta=safe_oracle,
        top5_delta=top5,
        component_deltas={
            name: np.zeros(count) for name in ("collision", "drivable", "ttc")
        },
        buckets=buckets,
        logs=[f"{folds[i]}:{noises[i]}:log{i // 2}" for i in range(count)],
        folds=folds,
        noises=noises,
        seed=34,
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


def test_stage34_trimmed_mean_discards_symmetric_tails():
    values = np.concatenate(([-100.0], np.ones(8), [100.0]))
    assert trimmed10_mean(values) == 1.0
