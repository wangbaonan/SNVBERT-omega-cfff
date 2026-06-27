from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "snvbert_phase_lattice_training_state_v1"


def contract_hash(contract: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_state(path: str | Path, model, optimizer, scheduler, stream, step: int, contract: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "schema": SCHEMA,
        "contract": contract,
        "contract_hash": contract_hash(contract),
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "stream": stream.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(payload, temporary)
    temporary.replace(target)


def load_state(path: str | Path, model, optimizer, scheduler, stream, contract: dict[str, Any]) -> int:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("contract_hash") != contract_hash(contract) or payload.get("contract") != contract:
        raise RuntimeError("checkpoint contract mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    stream.load_state_dict(payload["stream"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["step"])
