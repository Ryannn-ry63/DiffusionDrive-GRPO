from collections import Counter

from navsim.planning.training.grpo_sampler import (
    Stage30GlobalBucketBatchSampler,
)


def _data():
    tokens = []
    buckets = {}
    for bucket in range(4):
        for index in range(100):
            token = f"b{bucket}_{index}"
            tokens.append(token)
            buckets[token] = bucket
    return tokens, buckets


def _sampler(rank, seed=30):
    tokens, buckets = _data()
    return Stage30GlobalBucketBatchSampler(
        tokens, buckets, optimizer_steps_per_epoch=48,
        composition=(2, 30, 8, 24), accumulation_steps=8,
        world_size=8, rank=rank, seed=seed,
    )


def test_stage30_sampler_exact_global_mix_and_rank_disjointness():
    tokens, buckets = _data()
    shards = []
    for rank in range(8):
        values = [batch[0] for batch in list(_sampler(rank))[:8]]
        assert len(values) == len(set(values)) == 8
        shards.append(values)
    merged = [index for shard in shards for index in shard]
    assert len(merged) == len(set(merged)) == 64
    counts = Counter(buckets[tokens[index]] for index in merged)
    assert [counts[index] for index in range(4)] == [2, 30, 8, 24]


def test_stage30_sampler_branches_epochs_and_resume_are_deterministic():
    cov = _sampler(rank=3, seed=71)
    mcc = _sampler(rank=3, seed=71)
    first = list(cov)
    assert first == list(mcc)
    second_epoch = list(cov)
    assert first != second_epoch
    replay = _sampler(rank=3, seed=71)
    replay.set_epoch(1)
    assert second_epoch == list(replay)

    full = list(_sampler(rank=5, seed=99))
    resumed = _sampler(rank=5, seed=99)
    resumed.set_epoch(0, start_optimizer_step=11)
    assert full[11 * 8:] == list(resumed)


def test_stage30_sampler_rejects_wrong_world_or_sparse_bucket():
    tokens, buckets = _data()
    try:
        Stage30GlobalBucketBatchSampler(
            tokens, buckets, world_size=4, rank=0, accumulation_steps=8
        )
    except ValueError as error:
        assert "world_size" in str(error)
    else:
        raise AssertionError("wrong world size unexpectedly passed")
    sparse_tokens = [token for token in tokens if buckets[token] != 0][:100]
    try:
        Stage30GlobalBucketBatchSampler(
            sparse_tokens, buckets, world_size=8, rank=0,
        )
    except ValueError as error:
        assert "bucket 0" in str(error)
    else:
        raise AssertionError("empty severe bucket unexpectedly passed")

