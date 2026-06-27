from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from src.snvbert_mamba import (
    GlobalWindowStream,
    PackedWindowDataset,
    SNVBertMamba,
    SNVBertMambaConfig,
    TensorWindowDataset,
    collate_windows,
    build_optimizer,
    learning_rate_multiplier,
    save_state,
    train_updates,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="snvbert-phase-lattice")
    value.add_argument("--train-manifest", default="")
    value.add_argument("--train-pack-dir", default="")
    value.add_argument("--output", required=True)
    value.add_argument("--backend", choices=("metax_mamba2", "official_mamba3"), default="metax_mamba2")
    value.add_argument("--backend-identity", required=True)
    value.add_argument("--input-dim", type=int, default=512)
    value.add_argument("--feedforward-dim", type=int, default=4480)
    value.add_argument("--geometry", choices=("none", "film"), default="none")
    value.add_argument("--enable-distance-bias", action="store_true")
    value.add_argument("--global-batch", type=int, default=128)
    value.add_argument("--micro-batch", type=int, default=1)
    value.add_argument("--maximum-updates", type=int, default=81303)
    value.add_argument("--warmup-updates", type=int, default=4096)
    value.add_argument("--learning-rate", type=float, default=6.0e-5)
    value.add_argument("--minimum-rate", type=float, default=0.1)
    value.add_argument("--weight-decay", type=float, default=0.01)
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--seed", type=int, default=42)
    return value


def distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires an accelerator")
    torch.cuda.set_device(local)
    return world, rank, torch.device("cuda", local)


def main() -> None:
    args = parser().parse_args()
    world, rank, device = distributed()
    torch.manual_seed(args.seed)
    if bool(args.train_manifest) == bool(args.train_pack_dir):
        raise ValueError("provide exactly one training data source")
    dataset = TensorWindowDataset(args.train_manifest) if args.train_manifest else PackedWindowDataset(args.train_pack_dir, seed=args.seed)
    if args.global_batch % (world * args.micro_batch):
        raise ValueError("global batch must divide world size times micro batch")
    accumulation = args.global_batch // (world * args.micro_batch)
    stream = GlobalWindowStream(len(dataset), args.global_batch, world, rank, args.seed)
    loader = DataLoader(dataset, batch_size=args.micro_batch, sampler=stream, num_workers=args.workers, collate_fn=collate_windows, pin_memory=True)
    config = SNVBertMambaConfig(
        input_dim=args.input_dim,
        feedforward_dim=args.feedforward_dim,
        geometry_mode=args.geometry,
        distance_bias=bool(args.enable_distance_bias),
        backend=args.backend,
        backend_identity=args.backend_identity,
        state_chunk=64 if args.backend == "official_mamba3" else 256,
    )
    model = SNVBertMamba(config).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    optimizer, optimizer_audit = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step,
            args.warmup_updates,
            args.maximum_updates,
            args.minimum_rate,
        ),
    )
    result = train_updates(model, loader, optimizer, scheduler, device, accumulation, args.maximum_updates, 1.0)
    stream.cursor = int(result["updates"])
    target = model.module if hasattr(model, "module") else model
    contract = {
        "schema": "snvbert_phase_lattice_training_contract_v1",
        "config": asdict(config),
        "architecture": target.backbone.architecture_record(),
        "data_source": str(Path(args.train_manifest or args.train_pack_dir).resolve()),
        "global_batch": args.global_batch,
        "micro_batch": args.micro_batch,
        "accumulation": accumulation,
        "world": world,
        "seed": args.seed,
        "maximum_updates": args.maximum_updates,
        "optimizer": optimizer_audit,
    }
    if rank == 0:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        save_state(output / "snvbert-phase-lattice.pt", target, optimizer, scheduler, stream, int(result["updates"]), contract)
        (output / "training-result.json").write_text(json.dumps({**result, "contract": contract}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
