from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backend import StateEngineSpec, StateFactory, create_state_engine
from .config import SNVBertMambaConfig
from .geometry import DirectionalGeometry
from .segments import SegmentLayout


def masked(tensor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid.unsqueeze(-1), tensor, torch.zeros((), device=tensor.device, dtype=tensor.dtype))


class RootMeanSquareNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = tensor.float().square().mean(-1, keepdim=True).add(self.epsilon).rsqrt()
        return (tensor.float() * scale * self.weight.float()).to(tensor.dtype)


class GeometryModulation(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2 * hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tensor: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        scale, shift = self.projection(geometry.to(tensor.dtype)).chunk(2, dim=-1)
        return tensor * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift


class LocalSwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, feedforward_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feedforward_dim = feedforward_dim
        self.expand = nn.Linear(hidden_dim, 2 * feedforward_dim)
        self.local = nn.Conv1d(feedforward_dim, feedforward_dim, 5, padding=2, groups=feedforward_dim)
        self.contract = nn.Linear(feedforward_dim, hidden_dim)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def forward(self, tensor: torch.Tensor, layout: SegmentLayout) -> torch.Tensor:
        if layout.token_count == 0:
            return torch.zeros_like(tensor)
        projected = self.expand(tensor).reshape(-1, 2 * self.feedforward_dim)
        result = tensor.new_zeros((layout.token_count, self.feedforward_dim))
        for _, offsets, source in layout.groups:
            packed = projected.index_select(0, source.reshape(-1)).reshape(source.shape[0], source.shape[1], -1)
            value, gate = packed.chunk(2, dim=-1)
            local = self.local(value.transpose(1, 2)).transpose(1, 2)
            activated = F.silu(value + local) * gate
            result = result.index_copy(
                0,
                offsets.reshape(-1),
                activated.reshape(-1, self.feedforward_dim).to(dtype=result.dtype),
            )
        return self.contract(layout.scatter(result))


class PairExchange(nn.Module):
    def __init__(self, hidden_dim: int, exchange_dim: int) -> None:
        super().__init__()
        self.mean_up = nn.Linear(hidden_dim, exchange_dim)
        self.difference_up = nn.Linear(hidden_dim, exchange_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, exchange_dim)
        self.mean_down = nn.Linear(exchange_dim, hidden_dim)
        self.difference_down = nn.Linear(exchange_dim, hidden_dim, bias=False)

    def forward(self, tensor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe = masked(tensor.reshape(-1, tensor.shape[2], tensor.shape[3]), valid.reshape(-1, valid.shape[2])).reshape_as(tensor)
        mean = 0.5 * (safe[:, 0] + safe[:, 1])
        difference = 0.5 * (safe[:, 0] - safe[:, 1])
        mean_hidden = F.silu(self.mean_up(mean))
        difference_hidden = torch.tanh(self.difference_up(difference))
        mean_delta = self.mean_down(mean_hidden + difference_hidden.square())
        difference_delta = self.difference_down(difference_hidden * torch.sigmoid(self.gate(mean)))
        output = torch.stack((mean_delta + difference_delta, mean_delta - difference_delta), dim=1)
        return masked(output.reshape(-1, output.shape[2], output.shape[3]), valid.reshape(-1, valid.shape[2])).reshape_as(output)


class BidirectionalState(nn.Module):
    def __init__(self, config: SNVBertMambaConfig, index: int, spec: StateEngineSpec, factory: Optional[StateFactory]) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.state_dim = config.state_dim
        self.input_projection = nn.Linear(config.hidden_dim, config.state_dim, bias=False)
        common = {
            "spec": spec,
            "width": config.state_dim,
            "state_size": config.state_size,
            "expand": config.state_expand,
            "head_dim": config.state_head_dim,
            "chunk": config.state_chunk,
            "convolution": config.state_conv,
            "factory": factory,
        }
        self.forward_engine = create_state_engine(layer_index=2 * index, direction="forward", **common)
        self.backward_engine = create_state_engine(layer_index=2 * index + 1, direction="backward", **common)
        self.fusion_norm = RootMeanSquareNorm(2 * config.state_dim, config.norm_epsilon)
        self.output_projection = nn.Linear(2 * config.state_dim, config.hidden_dim)
        self.geometry = GeometryModulation(config.hidden_dim) if config.geometry_mode == "film" else None

    def forward(self, tensor: torch.Tensor, layout: SegmentLayout, geometry: Optional[DirectionalGeometry]) -> torch.Tensor:
        if layout.token_count == 0:
            return torch.zeros_like(tensor)
        forward_input = tensor
        backward_input = tensor
        if self.geometry is not None:
            if geometry is None:
                raise ValueError("geometry is required")
            left = geometry.left.repeat_interleave(2, dim=0).to(tensor.device)
            right = geometry.right.repeat_interleave(2, dim=0).to(tensor.device)
            forward_input = self.geometry(tensor, left)
            backward_input = self.geometry(tensor, right)
        forward_flat = self.input_projection(forward_input).reshape(-1, self.state_dim)
        backward_flat = self.input_projection(backward_input).reshape(-1, self.state_dim)
        result = tensor.new_zeros((layout.token_count, 2 * self.state_dim))
        for _, offsets, source in layout.groups:
            shape = (source.shape[0], source.shape[1], self.state_dim)
            left_input = forward_flat.index_select(0, source.reshape(-1)).reshape(shape)
            right_input = backward_flat.index_select(0, source.reshape(-1)).reshape(shape)
            left_output = self.forward_engine(left_input)
            right_output = torch.flip(self.backward_engine(torch.flip(right_input, (1,))), (1,))
            fused = torch.cat((left_output, right_output), dim=-1)
            result = result.index_copy(
                0,
                offsets.reshape(-1),
                fused.reshape(-1, 2 * self.state_dim).to(dtype=result.dtype),
            )
        return self.output_projection(self.fusion_norm(layout.scatter(result)))


class PhysicalAttention(nn.Module):
    def __init__(self, config: SNVBertMambaConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.heads = config.attention_heads
        self.head_dim = config.hidden_dim // config.attention_heads
        self.query_key_value = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.query_weight = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.key_weight = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.norm_epsilon = float(config.norm_epsilon)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        frequency = 1.0 / (config.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("frequency", frequency, persistent=False)
        self.rope_scale = config.rope_scale
        self.distance_limit = config.distance_limit_bp
        self.distance_buckets = config.distance_buckets
        self.distance_table = nn.Embedding(config.distance_buckets, config.attention_heads) if config.distance_bias else None
        if self.distance_table is not None:
            nn.init.zeros_(self.distance_table.weight)

    @staticmethod
    def _rotate(tensor: torch.Tensor) -> torch.Tensor:
        first = tensor[..., ::2]
        second = tensor[..., 1::2]
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _head_norm(self, tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] != weight.shape:
            raise ValueError("attention head shape mismatch")
        inverse = tensor.float().square().mean(dim=-1, keepdim=True).add(self.norm_epsilon).rsqrt()
        return (tensor.float() * inverse * weight.float()).to(dtype=tensor.dtype)

    def forward(self, tensor: torch.Tensor, positions: torch.Tensor, layout: SegmentLayout) -> torch.Tensor:
        if layout.token_count == 0:
            return torch.zeros_like(tensor)
        flat_tensor = tensor.reshape(-1, self.hidden_dim)
        flat_positions = positions.reshape(-1)
        result = tensor.new_zeros((layout.token_count, self.hidden_dim))
        for _, offsets, source in layout.groups:
            packed = flat_tensor.index_select(0, source.reshape(-1)).reshape(source.shape[0], source.shape[1], self.hidden_dim)
            packed_positions = flat_positions.index_select(0, source.reshape(-1)).reshape(source.shape)
            qkv = self.query_key_value(packed).reshape(packed.shape[0], packed.shape[1], 3, self.heads, self.head_dim)
            query = self._head_norm(qkv[:, :, 0], self.query_weight)
            key = self._head_norm(qkv[:, :, 1], self.key_weight)
            value = qkv[:, :, 2]
            relative = packed_positions - packed_positions[:, :1]
            angle = relative.float().unsqueeze(-1) / self.rope_scale * self.frequency
            cosine = torch.repeat_interleave(angle.cos(), 2, dim=-1).unsqueeze(2).to(query.dtype)
            sine = torch.repeat_interleave(angle.sin(), 2, dim=-1).unsqueeze(2).to(query.dtype)
            query = query * cosine + self._rotate(query) * sine
            key = key * cosine + self._rotate(key) * sine
            bias = None
            if self.distance_table is not None:
                distance = (packed_positions.unsqueeze(2) - packed_positions.unsqueeze(1)).abs().clamp_max(self.distance_limit)
                bucket = torch.floor(torch.log1p(distance.float()) / math.log1p(self.distance_limit) * (self.distance_buckets - 1)).long()
                bias = self.distance_table(bucket).permute(0, 3, 1, 2).to(query.dtype)
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=bias,
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape_as(packed)
            result = result.index_copy(
                0,
                offsets.reshape(-1),
                self.output(attended).reshape(-1, self.hidden_dim).to(dtype=result.dtype),
            )
        return layout.scatter(result)


class StateLayer(nn.Module):
    def __init__(self, config: SNVBertMambaConfig, index: int, spec: StateEngineSpec, factory: Optional[StateFactory]) -> None:
        super().__init__()
        self.state_norm = RootMeanSquareNorm(config.hidden_dim, config.norm_epsilon)
        self.state = BidirectionalState(config, index, spec, factory)
        self.state_scale = nn.Parameter(torch.tensor(0.1))
        self.local_norm = RootMeanSquareNorm(config.hidden_dim, config.norm_epsilon)
        self.local = LocalSwiGLU(config.hidden_dim, config.feedforward_dim)

    def forward(self, tensor: torch.Tensor, layout: SegmentLayout, geometry: Optional[DirectionalGeometry]) -> torch.Tensor:
        tensor = tensor + self.state_scale.to(tensor.dtype) * self.state(self.state_norm(tensor), layout, geometry)
        return tensor + self.local(self.local_norm(tensor), layout)


class AttentionLayer(nn.Module):
    def __init__(self, config: SNVBertMambaConfig) -> None:
        super().__init__()
        self.attention_norm = RootMeanSquareNorm(config.hidden_dim, config.norm_epsilon)
        self.attention = PhysicalAttention(config)
        self.attention_scale = nn.Parameter(torch.tensor(0.1))
        self.local_norm = RootMeanSquareNorm(config.hidden_dim, config.norm_epsilon)
        self.local = LocalSwiGLU(config.hidden_dim, config.feedforward_dim)
        self.exchange_norm = RootMeanSquareNorm(config.hidden_dim, config.norm_epsilon)
        self.exchange = PairExchange(config.hidden_dim, config.exchange_dim)
        self.exchange_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, tensor: torch.Tensor, paired_shape: tuple[int, int, int, int], valid: torch.Tensor, positions: torch.Tensor, layout: SegmentLayout) -> torch.Tensor:
        tensor = tensor + self.attention_scale.to(tensor.dtype) * self.attention(self.attention_norm(tensor), positions, layout)
        tensor = tensor + self.local(self.local_norm(tensor), layout)
        paired = tensor.reshape(paired_shape)
        paired = paired + self.exchange_scale.to(tensor.dtype) * self.exchange(self.exchange_norm(paired), valid)
        return paired.reshape_as(tensor)
