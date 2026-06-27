from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backend import StateEngineSpec, StateFactory
from .config import SNVBertMambaConfig
from .geometry import directional_geometry
from .layers import AttentionLayer, StateLayer, masked
from .segments import SegmentLayout


class InterleavedGenomeBackbone(nn.Module):
    def __init__(self, config: SNVBertMambaConfig, spec: StateEngineSpec, factory: Optional[StateFactory] = None) -> None:
        super().__init__()
        self.config = config
        self.pattern = ("S", "S", "A") * config.stages
        layers = []
        state_index = 0
        for kind in self.pattern:
            if kind == "S":
                layers.append(StateLayer(config, state_index, spec, factory))
                state_index += 1
            else:
                layers.append(AttentionLayer(config))
        self.layers = nn.ModuleList(layers)
        self.spec = spec

    def forward(self, paired: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor, segments: Optional[torch.Tensor]) -> torch.Tensor:
        if paired.dim() != 4 or paired.shape[1] != 2:
            raise ValueError("paired state must be [batch,2,length,hidden]")
        batch, _, loci, hidden = paired.shape
        if hidden != self.config.hidden_dim or valid.shape != (batch, 2, loci) or positions.shape != (batch, loci):
            raise ValueError("backbone input shape mismatch")
        flat_valid = valid.reshape(2 * batch, loci)
        flat_segments = None if segments is None else segments.unsqueeze(1).expand(-1, 2, -1).reshape(2 * batch, loci)
        layout = SegmentLayout.build(flat_valid, flat_segments)
        flat = masked(paired.reshape(2 * batch, loci, hidden), flat_valid)
        flat_positions = positions.unsqueeze(1).expand(-1, 2, -1).reshape(2 * batch, loci)
        geometry = None
        if self.config.geometry_mode == "film":
            geometry = directional_geometry(positions, valid[:, 0], segments)
        shape = paired.shape
        for kind, layer in zip(self.pattern, self.layers):
            if kind == "S":
                flat = layer(flat, layout, geometry)
            else:
                flat = layer(flat, shape, valid, flat_positions, layout)
            flat = masked(flat, flat_valid)
        return flat.reshape(shape)

    def architecture_record(self) -> dict[str, object]:
        return {
            "schema": "snvbert_phase_lattice_architecture_v1",
            "name": "SNVBERT PhaseLattice",
            "pattern": list(self.pattern),
            "state_layers": 8,
            "attention_layers": 4,
            "exchange_layers": 4,
            "local_operator": "in_block_depthwise_swiglu_k5",
            "phase_representation": "explicit_paired_stream",
            "state_engine": self.spec.as_record(),
        }
