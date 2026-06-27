from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _module_parameter_ids(model: nn.Module, predicate) -> set[int]:
    result = set()
    for module in model.modules():
        if predicate(module):
            result.update(id(parameter) for parameter in module.parameters(recurse=False))
    return result


def parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    embedding_ids = _module_parameter_ids(
        model,
        lambda module: isinstance(module, (nn.Embedding, nn.EmbeddingBag)),
    )
    norm_ids = _module_parameter_ids(
        model,
        lambda module: "norm" in type(module).__name__.casefold(),
    )
    decay = []
    zero_decay = []
    classifications = []
    seen = set()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError("trainable parameter object is registered more than once")
        seen.add(id(parameter))
        lowered = name.casefold()
        final = lowered.rsplit(".", 1)[-1]
        state_dynamic = final in {
            "a_log",
            "d",
            "dt_bias",
            "delta_bias",
            "b_bias",
            "c_bias",
        }
        explicit_zero = any(
            token in lowered
            for token in (
                "mask_state",
                "latent_queries",
                "query_weight",
                "key_weight",
                "kernel",
                "log_width",
            )
        )
        scalar_control = parameter.ndim <= 1 and any(
            token in lowered for token in ("scale", "gate")
        )
        no_decay = (
            id(parameter) in embedding_ids
            or id(parameter) in norm_ids
            or parameter.ndim < 2
            or final == "bias"
            or state_dynamic
            or explicit_zero
            or scalar_control
        )
        target = zero_decay if no_decay else decay
        target.append(parameter)
        classifications.append(
            {
                "name": name,
                "policy": "zero_decay" if no_decay else "decay",
                "numel": int(parameter.numel()),
            }
        )
    if len(seen) != len(classifications):
        raise RuntimeError("optimizer parameter coverage is not unique")
    groups = [
        {"name": "matrix_decay", "params": decay, "weight_decay": float(weight_decay)},
        {"name": "structural_zero_decay", "params": zero_decay, "weight_decay": 0.0},
    ]
    audit = {
        "schema": "snvbert_phase_lattice_optimizer_groups_v1",
        "trainable_tensor_count": len(classifications),
        "trainable_parameter_count": sum(item["numel"] for item in classifications),
        "all_parameters_classified_once": True,
        "groups": [
            {
                "name": group["name"],
                "weight_decay": group["weight_decay"],
                "tensor_count": len(group["params"]),
                "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
            }
            for group in groups
        ],
        "classifications": classifications,
    }
    return groups, audit


def learning_rate_multiplier(
    step: int,
    warmup: int,
    total: int,
    minimum: float,
) -> float:
    if total <= 0 or warmup < 0 or warmup >= total:
        raise ValueError("invalid scheduler horizon")
    if not 0.0 <= float(minimum) <= 1.0:
        raise ValueError("minimum learning-rate ratio must be in [0,1]")
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = min(1.0, float(step - warmup + 1) / max(1, total - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum) + (1.0 - float(minimum)) * cosine


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    groups, audit = parameter_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    audit["learning_rate"] = float(learning_rate)
    audit["betas"] = [0.9, 0.95]
    audit["eps"] = 1.0e-8
    return optimizer, audit


__all__ = ["build_optimizer", "learning_rate_multiplier", "parameter_groups"]
