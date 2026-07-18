"""Deterministic uniform-plus-priority sampling for GRPO hard-case training."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

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
