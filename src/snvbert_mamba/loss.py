from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F


LOSS_KEYS = ("haploid", "diploid", "prior_window", "prior_token")


def _row_weights(
    labels: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    if sample_weight is None:
        return torch.ones(
            labels.shape[0],
            device=labels.device,
            dtype=torch.float32,
        )
    weights = sample_weight.to(device=labels.device, dtype=torch.float32)
    if weights.shape != (labels.shape[0],):
        raise ValueError("sample weights must provide one value per haplotype")
    if labels.shape[0] % 2 or not torch.equal(weights[0::2], weights[1::2]):
        raise ValueError("sample weights must be shared by paired phases")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("sample weights must be finite and nonnegative")
    return weights


def counted_loss_counts(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    if labels.dim() != 2 or labels.shape[0] % 2:
        raise ValueError("labels must contain flattened adjacent phase pairs")
    if attention_mask.shape != labels.shape:
        raise ValueError("attention mask must match labels")
    weights = _row_weights(labels, sample_weight).double()
    target = labels.ne(-100)
    paired_target = target.reshape(-1, 2, labels.shape[1]).all(dim=1)
    pair_weights = weights.reshape(-1, 2)[:, 0]
    row_has_context = attention_mask.bool().any(dim=1)
    return {
        "haploid": (target.double() * weights[:, None]).sum(),
        "diploid": (paired_target.double() * pair_weights[:, None]).sum(),
        "prior_window": (row_has_context.double() * weights).sum(),
        "prior_token": (
            attention_mask.bool().double() * weights[:, None]
        ).sum(),
    }


def counted_loss_parts(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    logits = outputs["logits"]
    attention = outputs["attention_mask"].bool()
    if logits.shape[:2] != labels.shape or logits.shape[-1] != 2:
        raise ValueError("binary logits must match labels")
    weights = _row_weights(labels, sample_weight).to(dtype=logits.dtype)
    target = labels.ne(-100)
    safe_target = labels.clamp_min(0).long()
    haploid_values = F.cross_entropy(
        logits.reshape(-1, 2),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    haploid_sum = (
        haploid_values * target.to(haploid_values.dtype) * weights[:, None]
    ).sum()

    paired_labels = labels.reshape(-1, 2, labels.shape[1])
    paired_target = paired_labels.ne(-100).all(dim=1)
    pair_weights = weights.reshape(-1, 2)[:, 0]
    probability = torch.sigmoid(logits[..., 1] - logits[..., 0]).reshape(
        -1,
        2,
        labels.shape[1],
    )
    first, second = probability[:, 0], probability[:, 1]
    genotype = torch.stack(
        (
            (1.0 - first) * (1.0 - second),
            first * (1.0 - second) + (1.0 - first) * second,
            first * second,
        ),
        dim=-1,
    ).clamp_min(1.0e-8)
    diploid_target = paired_labels.clamp_min(0).sum(dim=1).long()
    diploid_values = F.nll_loss(
        genotype.log().reshape(-1, 3),
        diploid_target.reshape(-1),
        reduction="none",
    ).reshape_as(paired_target)
    diploid_sum = (
        diploid_values
        * paired_target.to(diploid_values.dtype)
        * pair_weights[:, None]
    ).sum()

    window_values = 1.0 - F.cosine_similarity(
        outputs["window_summary"].float(),
        outputs["prior_target"].float(),
        dim=-1,
    )
    row_has_context = attention.any(dim=1)
    prior_window_sum = (
        window_values
        * row_has_context.to(window_values.dtype)
        * weights.to(window_values.dtype)
    ).sum()
    token_values = 1.0 - F.cosine_similarity(
        outputs["prior_alignment_states"].float(),
        outputs["prior_token_target"].float(),
        dim=-1,
    )
    prior_token_sum = (
        token_values
        * attention.to(token_values.dtype)
        * weights[:, None].to(token_values.dtype)
    ).sum()
    sums = {
        "haploid": haploid_sum,
        "diploid": diploid_sum,
        "prior_window": prior_window_sum,
        "prior_token": prior_token_sum,
    }
    return sums, counted_loss_counts(labels, attention, sample_weight)


def normalized_loss(
    local_sums: Mapping[str, torch.Tensor],
    global_counts: Mapping[str, torch.Tensor],
    world_size: int,
) -> torch.Tensor:
    if set(local_sums) != set(LOSS_KEYS) or set(global_counts) != set(LOSS_KEYS):
        raise ValueError("counted loss keys are incomplete")
    first = local_sums["haploid"]
    total = first * 0.0
    coefficients = {
        "haploid": 1.0,
        "diploid": 0.5,
        "prior_window": 0.02,
        "prior_token": 0.02,
    }
    for key in LOSS_KEYS:
        count = global_counts[key].to(device=first.device, dtype=first.dtype)
        term = torch.where(
            count.gt(0),
            local_sums[key] / count.clamp_min(1.0),
            local_sums[key] * 0.0,
        )
        total = total + float(world_size) * coefficients[key] * term
    return total


__all__ = [
    "LOSS_KEYS",
    "counted_loss_counts",
    "counted_loss_parts",
    "normalized_loss",
]
