from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class GlobalWindowStream(Sampler[int]):
    def __init__(self, size: int, global_batch: int, replicas: int, rank: int, seed: int = 42, cursor: int = 0, epoch: int = 0) -> None:
        self.size = int(size)
        self.global_batch = int(global_batch)
        self.replicas = int(replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.cursor = int(cursor)
        self.epoch = int(epoch)
        if self.size <= 0 or self.global_batch <= 0 or self.global_batch % self.replicas:
            raise ValueError("invalid stream dimensions")
        if not 0 <= self.rank < self.replicas:
            raise ValueError("invalid rank")
        self.local_batch = self.global_batch // self.replicas

    @property
    def updates(self) -> int:
        return math.ceil(self.size / self.global_batch)

    def permutation(self) -> torch.Tensor:
        return torch.randperm(
            self.size,
            generator=torch.Generator().manual_seed(self.seed + self.epoch),
        )

    def __iter__(self) -> Iterator[int]:
        permutation = self.permutation()
        padding = self.updates * self.global_batch - self.size
        if padding:
            permutation = torch.cat((permutation, torch.full((padding,), -1, dtype=torch.long)))
        matrix = permutation.reshape(self.updates, self.replicas, self.local_batch)
        yield from (int(value) for value in matrix[self.cursor :, self.rank].reshape(-1).tolist())

    def __len__(self) -> int:
        return (self.updates - self.cursor) * self.local_batch

    def state_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "global_batch": self.global_batch,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if any(int(state[key]) != getattr(self, key) for key in ("size", "global_batch", "seed")):
            raise RuntimeError("stream identity mismatch")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
