from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import SNVBertMambaConfig


class ResidualCrossUpdate(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.first_norm = nn.LayerNorm(width)
        self.second_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        safe_padding = None if memory_padding is None else memory_padding.bool()
        all_hidden = None
        if safe_padding is not None:
            all_hidden = safe_padding.all(dim=1)
            if bool(all_hidden.any()):
                safe_padding = safe_padding.clone()
                safe_padding[all_hidden] = False
        attended = self.attention(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=safe_padding,
            need_weights=False,
        )[0]
        updated = self.first_norm(query + self.dropout(attended))
        updated = self.second_norm(updated + self.dropout(self.feedforward(updated)))
        if all_hidden is not None and bool(all_hidden.any()):
            updated = updated.masked_fill(all_hidden[:, None, None], 0.0)
        return updated


class LatentReconstruction(nn.Module):
    def __init__(self, config: SNVBertMambaConfig) -> None:
        super().__init__()
        width = config.hidden_dim
        self.latent_queries = nn.Parameter(
            torch.randn(config.latent_count, width) * 0.02
        )
        self.latent_update = ResidualCrossUpdate(
            width,
            config.attention_heads,
            config.decoder_feedforward_dim,
            config.dropout,
        )
        refine_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.attention_heads,
            dim_feedforward=config.decoder_feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.latent_refine = nn.TransformerEncoder(
            refine_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.decode_update = ResidualCrossUpdate(
            width,
            config.attention_heads,
            config.decoder_feedforward_dim,
            config.dropout,
        )
        self.output_norm = nn.LayerNorm(width, eps=config.layer_norm_epsilon)

    def forward(
        self,
        encoded: torch.Tensor,
        valid: torch.Tensor,
        decoder_spatial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.latent_queries.unsqueeze(0).expand(
            encoded.shape[0], -1, -1
        ).to(device=encoded.device, dtype=encoded.dtype)
        latents = self.latent_update(
            queries,
            encoded,
            memory_padding=~valid.bool(),
        )
        latents = self.latent_refine(latents)
        decoder_query = encoded + decoder_spatial.to(
            device=encoded.device,
            dtype=encoded.dtype,
        )
        decoded_latent = self.decode_update(decoder_query, latents)
        decoded = self.output_norm(decoded_latent + encoded)
        return decoded, latents, decoded_latent


__all__ = ["LatentReconstruction", "ResidualCrossUpdate"]
