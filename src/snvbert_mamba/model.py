from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .backend import StateFactory, resolve_state_engine
from .backbone import InterleavedGenomeBackbone
from .config import SNVBertMambaConfig
from .decoder import LatentReconstruction
from .diffusion import OrdinalLinkDiffusion


def _paired(value: torch.Tensor, rows: int, name: str) -> torch.Tensor:
    if rows <= 0 or rows % 2 or value.shape[0] != rows:
        raise ValueError(f"{name} row count must contain adjacent phase pairs")
    pair = value.reshape(rows // 2, 2, *value.shape[1:])
    mismatch = pair[:, 0] != pair[:, 1]
    if value.is_floating_point():
        mismatch &= ~(torch.isnan(pair[:, 0]) & torch.isnan(pair[:, 1]))
    if bool(mismatch.any()):
        raise ValueError(f"{name} must be shared by paired phases")
    return pair


def _masked_average(
    tensor: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    weight = valid.to(device=tensor.device, dtype=tensor.dtype).unsqueeze(-1)
    return (tensor * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class LearnedCoordinateEncoding(nn.Module):
    def __init__(self, config: SNVBertMambaConfig) -> None:
        super().__init__()
        width = config.hidden_dim
        self.megabase = nn.Embedding(config.megabase_bins, width)
        self.kilobase = nn.Embedding(config.kilobase_bins, width)
        self.base = nn.Embedding(config.base_bins, width)
        self.chromosome = nn.Embedding(
            config.chromosome_bins,
            width,
            padding_idx=0,
        )
        self.megabase_bins = int(config.megabase_bins)
        self.kilobase_bins = int(config.kilobase_bins)
        self.base_bins = int(config.base_bins)
        self.chromosome_bins = int(config.chromosome_bins)

    def forward(
        self,
        positions: torch.Tensor,
        chromosomes: torch.Tensor,
    ) -> torch.Tensor:
        position = positions.long().clamp_min(0)
        megabase = (position // 1_000_000).clamp(0, self.megabase_bins - 1)
        kilobase = ((position % 1_000_000) // 1000).clamp(
            0,
            self.kilobase_bins - 1,
        )
        base = (position % 1000).clamp(0, self.base_bins - 1)
        chromosome = chromosomes.long().clamp(0, self.chromosome_bins - 1)
        return (
            self.megabase(megabase)
            + self.kilobase(kilobase)
            + self.base(base)
            + self.chromosome(chromosome)
        )


class SNVBertMamba(nn.Module):
    def __init__(
        self,
        config: SNVBertMambaConfig,
        factory: Optional[StateFactory] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.engine = resolve_state_engine(config.backend, config.backend_identity)
        self.genotype = nn.Embedding(2, config.hidden_dim)
        self.mask_state = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.02,
                size=(1, 1, config.hidden_dim),
            )
        )
        self.allele_context = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.coordinate = LearnedCoordinateEncoding(config)
        self.window_context = nn.Sequential(
            nn.Linear(config.window_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.input_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.layer_norm_epsilon,
        )
        self.backbone = InterleavedGenomeBackbone(config, self.engine, factory)
        self.reconstruction = LatentReconstruction(config)
        self.summary_target = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.token_target = nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
        )
        self.diffusion = OrdinalLinkDiffusion(config)
        self.allele_head = nn.Linear(config.hidden_dim, 2)

    def _masked_genotype(
        self,
        genotypes: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        safe = genotypes.long().clamp(0, 1)
        observed = self.genotype(safe)
        neutral = self.genotype.weight.mean(dim=0).view(1, 1, -1)
        mix = float(self.config.hidden_genotype_blend)
        masked_state = (1.0 - mix) * self.mask_state + mix * neutral
        return torch.where(hidden.unsqueeze(-1), masked_state, observed)

    @staticmethod
    def _window_features(
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        position = positions.float()
        valid = valid.bool()
        count = valid.sum(dim=1)
        first_index = valid.long().argmax(dim=1, keepdim=True)
        ordinal = torch.arange(
            position.shape[1],
            device=position.device,
        ).view(1, -1)
        last_index = torch.where(
            valid,
            ordinal,
            torch.zeros_like(ordinal),
        ).max(dim=1, keepdim=True).values
        first = position.gather(1, first_index).squeeze(1)
        last = position.gather(1, last_index).squeeze(1)
        visible_fraction = count.float() / float(max(1, position.shape[1]))
        span = torch.log1p((last - first).clamp_min(0.0)) / math.log1p(
            250_000_000.0
        )
        return torch.stack((visible_fraction, span), dim=-1)

    def forward(
        self,
        *,
        ref_embeddings: torch.Tensor,
        alt_embeddings: torch.Tensor,
        genotypes: torch.Tensor,
        positions: torch.Tensor,
        chromosomes: torch.Tensor,
        valid_mask: torch.Tensor,
        hidden_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        rows, loci = genotypes.shape
        if rows <= 0 or rows % 2:
            raise ValueError("genotype rows must contain adjacent phase pairs")
        expected = (rows, loci, self.config.input_dim)
        if ref_embeddings.shape != expected or alt_embeddings.shape != expected:
            raise ValueError("allele embedding shape mismatch")
        pair_valid = _paired(valid_mask.bool(), rows, "valid_mask")
        pair_hidden = _paired(hidden_mask.bool(), rows, "hidden_mask")
        pair_prediction = _paired(
            prediction_mask.bool(),
            rows,
            "prediction_mask",
        )
        pair_positions = _paired(positions, rows, "positions")
        pair_chromosomes = _paired(chromosomes, rows, "chromosomes")
        pair_segments = (
            None
            if segment_ids is None
            else _paired(segment_ids, rows, "segment_ids")
        )
        if bool((pair_hidden & ~pair_valid).any()) or bool(
            (pair_prediction & ~pair_hidden).any()
        ):
            raise ValueError("prediction sites must be valid and hidden")

        hidden_mask = hidden_mask.bool()
        valid_mask = valid_mask.bool()
        contrast = alt_embeddings - ref_embeddings
        allele_state = self.allele_context(contrast.float()).to(
            dtype=self.genotype.weight.dtype
        )
        coordinate_state = self.coordinate(positions, chromosomes).to(
            dtype=self.genotype.weight.dtype
        )
        window_state = self.window_context(
            self._window_features(positions, valid_mask)
        ).to(dtype=self.genotype.weight.dtype)
        hidden = self.input_norm(
            self._masked_genotype(genotypes, hidden_mask)
            + allele_state
            + coordinate_state
            + window_state.unsqueeze(1)
        )
        hidden = torch.where(
            valid_mask.unsqueeze(-1),
            hidden,
            torch.zeros_like(hidden),
        )
        paired_hidden = hidden.reshape(
            rows // 2,
            2,
            loci,
            self.config.hidden_dim,
        )
        encoded_pair = self.backbone(
            paired_hidden,
            pair_valid,
            pair_positions[:, 0],
            None if pair_segments is None else pair_segments[:, 0],
        )
        encoded = encoded_pair.reshape(rows, loci, self.config.hidden_dim)
        decoded_base, latents, decoded_latent = self.reconstruction(
            encoded,
            valid_mask,
            coordinate_state,
        )
        refined = decoded_base + self.diffusion(decoded_base, valid_mask)
        logits = self.allele_head(refined)

        prior_summary = _masked_average(contrast, valid_mask)
        prior_target = self.summary_target(prior_summary.float()).to(
            dtype=encoded.dtype
        )
        prior_token_target = self.token_target(allele_state)
        window_summary = latents.mean(dim=1)

        alt_probability = torch.sigmoid(
            logits[..., 1] - logits[..., 0]
        ).reshape(rows // 2, 2, loci)
        first = alt_probability[:, 0]
        second = alt_probability[:, 1]
        phased = torch.stack(
            (
                (1.0 - first) * (1.0 - second),
                (1.0 - first) * second,
                first * (1.0 - second),
                first * second,
            ),
            dim=-1,
        )
        return {
            "logits": logits,
            "alt_probability": alt_probability,
            "phased_genotype_probability": phased,
            "dosage": first + second,
            "encoded": encoded,
            "decoded_base": decoded_base,
            "decoded": refined,
            "decoded_latent": decoded_latent,
            "latents": latents,
            "window_summary": window_summary,
            "prior_target": prior_target,
            "prior_alignment_states": encoded,
            "prior_token_target": prior_token_target,
            "attention_mask": valid_mask,
            "valid_mask": valid_mask,
            "prediction_mask": prediction_mask.bool(),
            "architecture": self.backbone.architecture_record(),
        }


__all__ = ["LearnedCoordinateEncoding", "SNVBertMamba"]
