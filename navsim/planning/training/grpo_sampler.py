"""Deterministic uniform-plus-priority sampling for GRPO hard-case training."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

import torch
from torch.utils.data import Sampler


def load_manifest_tokens(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        tokens = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        tokens = [record["token"] for record in payload["records"]]
    else:
        raise ValueError("priority manifest must be a list or contain records")
    tokens = [str(token) for token in tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError("priority manifest contains duplicate tokens")
    return tokens


def _draw_from_cycles(
    indices: Sequence[int], count: int, generator: torch.Generator
) -> list[int]:
    if count == 0:
        return []
    if not indices:
        raise ValueError("cannot sample from an empty index pool")
    output: list[int] = []
    index_tensor = torch.tensor(list(indices), dtype=torch.long)
    while len(output) < count:
        permutation = torch.randperm(len(index_tensor), generator=generator)
        output.extend(index_tensor[permutation].tolist())
    return output[:count]


class BalancedPriorityBatchSampler(Sampler[list[int]]):
    """Yield batches with fixed uniform and priority slot counts."""

    def __init__(
        self,
        dataset_tokens: Sequence[str],
        priority_tokens: Sequence[str],
        batch_size: int,
        priority_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < priority_fraction < 1.0:
            raise ValueError("priority_fraction must be strictly between 0 and 1")
        exact_priority = batch_size * priority_fraction
        priority_per_batch = int(round(exact_priority))
        if abs(priority_per_batch - exact_priority) > 1e-9:
            raise ValueError(
                "batch_size * priority_fraction must be an integer"
            )
        common_per_batch = batch_size - priority_per_batch
        if priority_per_batch == 0 or common_per_batch == 0:
            raise ValueError("every balanced batch needs priority and uniform samples")

        token_to_index = {str(token): index for index, token in enumerate(dataset_tokens)}
        if len(token_to_index) != len(dataset_tokens):
            raise ValueError("dataset tokens contain duplicates")
        priority_set = set(map(str, priority_tokens))
        missing = sorted(priority_set - set(token_to_index))
        if missing:
            raise ValueError(
                f"{len(missing)} priority tokens are absent from training data; "
                f"first={missing[0]}"
            )
        priority_indices = [
            token_to_index[token] for token in dataset_tokens if token in priority_set
        ]
        common_indices = list(range(len(dataset_tokens)))
        if not priority_indices or not common_indices:
            raise ValueError("priority and common pools must both be non-empty")

        self.priority_indices = priority_indices
        self.common_indices = common_indices
        self.batch_size = batch_size
        self.priority_per_batch = priority_per_batch
        self.common_per_batch = common_per_batch
        self.seed = int(seed)
        self.num_batches = math.ceil(len(dataset_tokens) / batch_size)
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1
        priority = _draw_from_cycles(
            self.priority_indices,
            self.num_batches * self.priority_per_batch,
            generator,
        )
        common = _draw_from_cycles(
            self.common_indices,
            self.num_batches * self.common_per_batch,
            generator,
        )
        for batch_index in range(self.num_batches):
            start_p = batch_index * self.priority_per_batch
            start_c = batch_index * self.common_per_batch
            batch = (
                priority[start_p : start_p + self.priority_per_batch]
                + common[start_c : start_c + self.common_per_batch]
            )
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in order]


class Stage30GlobalBucketBatchSampler(Sampler[list[int]]):
    """Shard an exact global bucket mixture across DDP ranks and microbatches.

    One optimizer step is a deterministic global permutation of 64 distinct
    scene indices with bucket counts 2/30/8/24. Rank ``r`` consumes positions
    ``r, r+world_size, ...`` as eight singleton microbatches. This makes the
    mixture a property of the optimizer step rather than of any individual
    rank, and gives COV/MCC branches identical schedules for the same seed.
    """

    def __init__(
        self,
        dataset_tokens: Sequence[str],
        token_buckets: Mapping[str, int],
        optimizer_steps_per_epoch: int = 48,
        composition: Sequence[int] = (2, 30, 8, 24),
        accumulation_steps: int = 8,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if optimizer_steps_per_epoch <= 0 or accumulation_steps <= 0:
            raise ValueError("Stage30 steps and accumulation must be positive")
        composition = tuple(int(value) for value in composition)
        if len(composition) != 4 or any(value <= 0 for value in composition):
            raise ValueError("Stage30 requires four positive bucket counts")
        tokens = [str(token) for token in dataset_tokens]
        if len(tokens) != len(set(tokens)):
            raise ValueError("Stage30 dataset tokens contain duplicates")
        missing = [token for token in tokens if token not in token_buckets]
        if missing:
            raise ValueError(f"Stage30 bucket manifest lacks token {missing[0]}")
        pools = [[] for _ in range(4)]
        for index, token in enumerate(tokens):
            bucket = int(token_buckets[token])
            if bucket not in range(4):
                raise ValueError(f"Stage30 invalid bucket for token {token}")
            pools[bucket].append(index)
        for bucket, (pool, required) in enumerate(zip(pools, composition)):
            if len(pool) < required:
                raise ValueError(
                    f"Stage30 bucket {bucket} has {len(pool)} scenes, needs {required}"
                )

        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.world_size = int(env_world if world_size is None else world_size)
        self.rank = int(env_rank if rank is None else rank)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Stage30 DDP rank/world size is invalid")
        global_size = sum(composition)
        if global_size != self.world_size * int(accumulation_steps):
            raise ValueError(
                "Stage30 global composition must equal world_size * accumulation"
            )
        self.pools = tuple(tuple(pool) for pool in pools)
        self.composition = composition
        self.optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        self.accumulation_steps = int(accumulation_steps)
        self.seed = int(seed)
        self._epoch = 0
        self._start_optimizer_step = 0

    def __len__(self) -> int:
        return (
            self.optimizer_steps_per_epoch - self._start_optimizer_step
        ) * self.accumulation_steps

    def set_epoch(self, epoch: int, start_optimizer_step: int = 0) -> None:
        if epoch < 0 or not 0 <= start_optimizer_step < self.optimizer_steps_per_epoch:
            raise ValueError("Stage30 sampler epoch/resume step is invalid")
        self._epoch = int(epoch)
        self._start_optimizer_step = int(start_optimizer_step)

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "start_optimizer_step": self._start_optimizer_step,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self.set_epoch(
            int(state["epoch"]), int(state.get("start_optimizer_step", 0))
        )

    def _global_step_indices(self, epoch: int, step: int) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + 1_000_003 * int(epoch) + 10_007 * int(step)
        )
        selected: list[int] = []
        for pool, count in zip(self.pools, self.composition):
            order = torch.randperm(len(pool), generator=generator)[:count]
            selected.extend(pool[index] for index in order.tolist())
        order = torch.randperm(len(selected), generator=generator).tolist()
        result = [selected[index] for index in order]
        if len(result) != len(set(result)):
            raise RuntimeError("Stage30 global optimizer step contains duplicates")
        return result

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self._epoch
        start = self._start_optimizer_step
        self._epoch += 1
        self._start_optimizer_step = 0
        for step in range(start, self.optimizer_steps_per_epoch):
            global_indices = self._global_step_indices(epoch, step)
            local = global_indices[self.rank :: self.world_size]
            if len(local) != self.accumulation_steps:
                raise RuntimeError("Stage30 rank shard has the wrong microbatch count")
            for index in local:
                yield [index]
