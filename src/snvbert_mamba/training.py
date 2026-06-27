from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist

from .loss import LOSS_KEYS, counted_loss_counts, counted_loss_parts, normalized_loss


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_updates(model, batches: Iterable[dict[str, torch.Tensor]], optimizer, scheduler, device: torch.device, accumulation: int, maximum_updates: int, clip: float = 1.0) -> dict[str, float]:
    world = dist.get_world_size() if dist.is_initialized() else 1
    iterator = iter(batches)
    completed = 0
    loss_total = 0.0
    model.train()
    while completed < maximum_updates:
        group = []
        for _ in range(accumulation):
            try:
                group.append(move_batch(next(iterator), device))
            except StopIteration:
                break
        if not group:
            break
        counts = torch.zeros(len(LOSS_KEYS), device=device, dtype=torch.float64)
        for batch in group:
            local_counts = counted_loss_counts(
                batch["labels"],
                batch["valid_mask"],
                batch.get("sample_weight"),
            )
            counts += torch.stack([local_counts[key] for key in LOSS_KEYS]).to(device)
        if dist.is_initialized():
            dist.all_reduce(counts)
        global_counts = {key: counts[index] for index, key in enumerate(LOSS_KEYS)}
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for index, batch in enumerate(group):
            sample_weight = batch.pop("sample_weight", None)
            labels = batch.pop("labels")
            context = model.no_sync() if hasattr(model, "no_sync") and index + 1 < len(group) else torch.enable_grad()
            with context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model(**batch)
                    sums, _ = counted_loss_parts(outputs, labels, sample_weight)
                    loss = normalized_loss(sums, global_counts, world)
                loss.backward()
                update_loss += float(loss.detach()) / world
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        completed += 1
        loss_total += update_loss
    return {"updates": float(completed), "mean_loss": loss_total / max(1, completed)}
