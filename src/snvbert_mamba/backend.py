from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch.nn as nn


StateFactory = Callable[..., nn.Module]


@dataclass(frozen=True)
class StateEngineSpec:
    name: str
    version: str
    identity: str

    def as_record(self) -> dict[str, str]:
        return asdict(self)


def resolve_state_engine(name: str, identity: str) -> StateEngineSpec:
    if name == "test":
        return StateEngineSpec(name, "test", identity)
    try:
        version = importlib.metadata.version("mamba-ssm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("mamba-ssm is not installed") from exc
    if name == "metax_mamba2" and "+metax" not in version.casefold():
        raise RuntimeError("the selected backend is not the registered MetaX distribution")
    return StateEngineSpec(name, version, identity)


def create_state_engine(
    spec: StateEngineSpec,
    *,
    width: int,
    state_size: int,
    expand: int,
    head_dim: int,
    chunk: int,
    convolution: int,
    layer_index: int,
    direction: str,
    factory: Optional[StateFactory] = None,
) -> nn.Module:
    common = {
        "d_model": width,
        "d_state": state_size,
        "expand": expand,
        "headdim": head_dim,
        "chunk_size": chunk,
        "layer_idx": layer_index,
    }
    if spec.name == "test":
        if factory is None:
            raise RuntimeError("test backend requires a factory")
        module = factory(direction=direction, **common)
    elif spec.name == "metax_mamba2":
        from mamba_ssm import Mamba2

        module = Mamba2(
            **common,
            d_conv=convolution,
            use_mem_eff_path=True,
        )
    else:
        if chunk != 64:
            raise RuntimeError("official Mamba3 requires chunk 64")
        try:
            from mamba_ssm import Mamba3
        except ImportError:
            from mamba_ssm.modules.mamba3 import Mamba3
        module = Mamba3(**common, is_mimo=False, mimo_rank=1, is_outproj_norm=False)
    if not isinstance(module, nn.Module):
        raise TypeError("state factory must return a module")
    return module

