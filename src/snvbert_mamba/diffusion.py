from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SNVBertMambaConfig
from .layers import masked


class OrdinalLinkDiffusion(nn.Module):
    def __init__(self, config: SNVBertMambaConfig) -> None:
        super().__init__()
        width = config.hidden_dim
        self.radius = int(config.diffusion_radius)
        self.basis_count = int(config.diffusion_basis)
        self.max_log_distance = float(config.diffusion_max_log_distance)
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, width, bias=False)
        self.kernel = nn.Parameter(torch.empty(2, width, self.basis_count))
        nn.init.normal_(self.kernel, std=0.02)
        centers = torch.linspace(0.0, self.max_log_distance, self.basis_count)
        self.register_buffer("centers", centers, persistent=True)
        spacing = self.max_log_distance / max(1, self.basis_count - 1)
        self.log_width = nn.Parameter(torch.tensor(float(spacing)).log())
        self.channel_mix = nn.Linear(width, width)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _neighbor(
        tensor: torch.Tensor,
        valid: torch.Tensor,
        *,
        offset: int,
        direction: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = torch.zeros_like(tensor)
        shifted_valid = torch.zeros_like(valid)
        if direction > 0:
            shifted[:, :-offset] = tensor[:, offset:]
            shifted_valid[:, :-offset] = valid[:, offset:]
        else:
            shifted[:, offset:] = tensor[:, :-offset]
            shifted_valid[:, offset:] = valid[:, :-offset]
        return shifted, shifted_valid

    def forward(self, tensor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.bool()
        center = self.value(self.norm(masked(tensor, valid)))
        aggregate = torch.zeros_like(center)
        width = F.softplus(self.log_width).clamp_min(1.0e-3)
        for offset in range(1, self.radius + 1):
            ordinal_distance = tensor.new_full(
                tensor.shape[:2],
                math.log1p(float(offset)),
            )
            radial = torch.exp(
                -0.5
                * (
                    (ordinal_distance.unsqueeze(-1) - self.centers.to(ordinal_distance))
                    / width.to(ordinal_distance)
                ).square()
            )
            for direction_index, direction in enumerate((-1, 1)):
                neighbor, neighbor_valid = self._neighbor(
                    center,
                    valid,
                    offset=offset,
                    direction=direction,
                )
                edge = torch.einsum(
                    "blm,dm->bld",
                    radial.to(dtype=center.dtype),
                    self.kernel[direction_index].to(dtype=center.dtype),
                )
                active = valid & neighbor_valid
                message = (neighbor - center) * edge
                aggregate = aggregate + torch.where(
                    active.unsqueeze(-1),
                    message,
                    torch.zeros_like(message),
                )
        aggregate = aggregate / math.sqrt(float(max(1, 2 * self.radius)))
        mixed = F.gelu(self.channel_mix(aggregate))
        return masked(self.output(mixed), valid)


__all__ = ["OrdinalLinkDiffusion"]
