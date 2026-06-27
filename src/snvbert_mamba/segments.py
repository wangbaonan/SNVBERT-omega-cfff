from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SegmentLayout:
    shape: tuple[int, int]
    flat_indices: torch.Tensor
    lengths: torch.Tensor
    groups: tuple[tuple[int, torch.Tensor, torch.Tensor], ...]

    @classmethod
    def build(cls, valid: torch.Tensor, segments: Optional[torch.Tensor]) -> "SegmentLayout":
        if valid.dim() != 2:
            raise ValueError("valid mask must be rank two")
        mask = valid.bool()
        if segments is not None and segments.shape != mask.shape:
            raise ValueError("segment identifiers must match the mask")
        rows, loci = mask.shape
        coordinates = mask.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            empty = torch.empty(0, device=mask.device, dtype=torch.long)
            return cls((rows, loci), empty, empty, ())
        row = coordinates[:, 0]
        locus = coordinates[:, 1]
        flat = row * loci + locus
        starts = torch.ones(flat.shape[0], device=mask.device, dtype=torch.bool)
        if flat.shape[0] > 1:
            starts[1:] = row[1:] != row[:-1]
            if segments is not None:
                values = segments[row, locus]
                starts[1:] |= values[1:] != values[:-1]
        group = starts.long().cumsum(0) - 1
        lengths = torch.bincount(group)
        offsets = lengths.cumsum(0) - lengths
        buckets = []
        for length_value in torch.unique(lengths, sorted=True).tolist():
            length = int(length_value)
            selected = lengths.eq(length).nonzero(as_tuple=False).flatten()
            token_offsets = offsets.index_select(0, selected).unsqueeze(1)
            token_offsets = token_offsets + torch.arange(length, device=mask.device).unsqueeze(0)
            source = flat.index_select(0, token_offsets.reshape(-1)).reshape(selected.numel(), length)
            buckets.append((length, token_offsets, source))
        return cls((rows, loci), flat, lengths, tuple(buckets))

    @property
    def token_count(self) -> int:
        return int(self.flat_indices.numel())

    def values(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3 or tensor.shape[:2] != self.shape:
            raise ValueError("tensor does not match layout")
        return tensor.reshape(-1, tensor.shape[-1]).index_select(0, self.flat_indices)

    def scatter(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() != 2 or values.shape[0] != self.token_count:
            raise ValueError("one value is required for every valid token")
        output = values.new_zeros((self.shape[0] * self.shape[1], values.shape[-1]))
        if self.token_count:
            output = output.index_copy(0, self.flat_indices, values)
        return output.reshape(self.shape[0], self.shape[1], values.shape[-1])

