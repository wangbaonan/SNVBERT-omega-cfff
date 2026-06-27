from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class BenchmarkRequest:
    batch: Mapping[str, torch.Tensor]
    split: str
    missing_rate: float
    mask_seed: int
    selection_identity: str

    def __post_init__(self) -> None:
        if self.split != "val":
            raise ValueError("standalone benchmark requests are validation-only")
        if not 0.0 <= float(self.missing_rate) <= 1.0:
            raise ValueError("missing rate must be in [0,1]")
        if not str(self.selection_identity).strip():
            raise ValueError("selection identity must be explicit")


@dataclass(frozen=True)
class BenchmarkResponse:
    comparator_id: str
    metrics: Mapping[str, float]
    example_count: int
    output_identity: str

    def __post_init__(self) -> None:
        if not str(self.comparator_id).strip():
            raise ValueError("comparator id must be explicit")
        if int(self.example_count) <= 0:
            raise ValueError("example count must be positive")
        if not str(self.output_identity).strip():
            raise ValueError("output identity must be explicit")


@runtime_checkable
class ExternalBenchmarkPort(Protocol):
    @property
    def comparator_id(self) -> str:
        ...

    def evaluate(self, request: BenchmarkRequest) -> BenchmarkResponse:
        ...


__all__ = ["BenchmarkRequest", "BenchmarkResponse", "ExternalBenchmarkPort"]
