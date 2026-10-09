"""Domain-alternating batches with reproducible per-sample augmentation on resume."""
from __future__ import annotations

import math
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler


def domain_at(step: int) -> str:
    return "coco" if step % 2 == 0 else "oem"


def consumed_updates(completed_steps: int, domain: str) -> int:
    if domain not in {"coco", "oem"}:
        raise ValueError("Unknown source domain")
    return (completed_steps + 1) // 2 if domain == "coco" else completed_steps // 2


class SeededSourceDataset(Dataset):
    def __init__(self, dataset, seed: int, rank: int):
        self.dataset, self.seed, self.rank = dataset, seed, rank

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        cycle, index = item
        # Existing paired transforms use the CPU Torch RNG. Isolate them from
        # model dropout, worker prefetch timing, and the number of model modules.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + cycle * 1000003 + index * 97 + self.rank)
            return self.dataset[index]


class SourceBatchSampler(Sampler):
    def __init__(self, dataset, *, rank, world, batch_size, seed, start_batch, batches):
        self.sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=False)
        self.batch_size, self.start_batch, self.batches = batch_size, start_batch, batches
        self.per_cycle = math.ceil(len(self.sampler) / batch_size)

    def __len__(self):
        return self.batches

    def __iter__(self):
        cached_cycle, indices = None, None
        for ordinal in range(self.start_batch, self.start_batch + self.batches):
            cycle, local_batch = divmod(ordinal, self.per_cycle)
            if cycle != cached_cycle:
                self.sampler.set_epoch(cycle)
                indices = list(self.sampler)
                cached_cycle = cycle
            start = local_batch * self.batch_size
            yield [(cycle, index) for index in indices[start:start + self.batch_size]]
