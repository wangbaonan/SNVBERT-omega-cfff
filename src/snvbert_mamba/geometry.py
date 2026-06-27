from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DirectionalGeometry:
    left: torch.Tensor
    right: torch.Tensor


def directional_geometry(
    positions: torch.Tensor,
    valid: torch.Tensor,
    segments: Optional[torch.Tensor],
    maximum_bp: int = 250_000_000,
) -> DirectionalGeometry:
    if positions.dim() != 2 or valid.shape != positions.shape:
        raise ValueError("position and valid tensors must share [batch,length]")
    if segments is not None and segments.shape != positions.shape:
        raise ValueError("segment identifiers must match positions")
    active = valid.bool()
    same = active[:, 1:] & active[:, :-1]
    if segments is not None:
        same &= segments[:, 1:] == segments[:, :-1]
    delta = positions[:, 1:].long() - positions[:, :-1].long()
    if bool((same & delta.lt(0)).any()):
        raise ValueError("positions must be monotonic inside each segment")
    delta = torch.where(same, delta.clamp(0, maximum_bp), torch.zeros_like(delta)).float()
    left_gap = torch.zeros_like(positions, dtype=torch.float32)
    right_gap = torch.zeros_like(positions, dtype=torch.float32)
    left_gap[:, 1:] = delta
    right_gap[:, :-1] = delta
    left_boundary = active.clone()
    right_boundary = active.clone()
    left_boundary[:, 1:] = active[:, 1:] & ~same
    right_boundary[:, :-1] = active[:, :-1] & ~same
    left = torch.stack((torch.log1p(left_gap), left_boundary.float()), dim=-1)
    right = torch.stack((torch.log1p(right_gap), right_boundary.float()), dim=-1)
    left = left * active.unsqueeze(-1)
    right = right * active.unsqueeze(-1)
    return DirectionalGeometry(left, right)

