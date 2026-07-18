from scripts.evaluation.build_grpo_dev_manifest import (
    ordered_token_sha256,
    stratified_hash_sample,
)


def test_stratified_hash_sample_is_deterministic_and_proportional():
    token_logs = {
        **{f"a{i}": "log_a" for i in range(6)},
        **{f"b{i}": "log_b" for i in range(3)},
        "c0": "log_c",
    }

    first = stratified_hash_sample(token_logs, limit=5, seed=17)
    second = stratified_hash_sample(token_logs, limit=5, seed=17)

    assert first == second
    assert len(first) == len(set(first)) == 5
    counts = {
        log_name: sum(token_logs[token] == log_name for token in first)
        for log_name in set(token_logs.values())
    }
    assert counts == {"log_a": 3, "log_b": 1, "log_c": 1}


def test_stratified_hash_sample_changes_order_with_seed():
    token_logs = {f"token{i}": "log" for i in range(20)}

    first = stratified_hash_sample(token_logs, limit=10, seed=1)
    second = stratified_hash_sample(token_logs, limit=10, seed=2)

    assert first != second
    assert ordered_token_sha256(first) != ordered_token_sha256(second)


def test_stratified_hash_sample_rejects_invalid_limit():
    token_logs = {"a": "log"}

    for limit in (0, 2):
        try:
            stratified_hash_sample(token_logs, limit=limit, seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"limit={limit} should fail")
