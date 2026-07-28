import pytest

from scripts.evaluation.audit_grpo_stage35_checkpoint import require


def test_stage35_override_audit_accepts_equivalent_numeric_renderings():
    require(
        {
            "float": "0.10",
            "high": "0.90",
            "list": "[2, 30, 8, 24]",
            "flag": "true",
            "path": "/frozen/input.ckpt",
        },
        {
            "float": "0.1",
            "high": "0.9",
            "list": "[2,30,8,24]",
            "flag": "true",
            "path": "/frozen/input.ckpt",
        },
    )


def test_stage35_override_audit_still_rejects_numeric_or_path_drift():
    with pytest.raises(RuntimeError, match="weight"):
        require({"weight": "0.11"}, {"weight": "0.1"})
    with pytest.raises(RuntimeError, match="checkpoint"):
        require(
            {"checkpoint": "/wrong/input.ckpt"},
            {"checkpoint": "/frozen/input.ckpt"},
        )
