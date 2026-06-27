from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.snvbert_mamba import (
    BenchmarkRequest,
    ExternalBenchmarkPort,
    GlobalWindowStream,
    SNVBertMamba,
    SNVBertMambaConfig,
)
from src.snvbert_mamba.loss import counted_loss_parts, normalized_loss
from src.snvbert_mamba.optim import parameter_groups


class FakeState(nn.Module):
    calls = 0

    def __init__(self, d_model: int, direction: str, **unused) -> None:
        super().__init__()
        self.direction = direction
        self.scale = nn.Parameter(torch.linspace(0.9, 1.1, d_model))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        type(self).calls += 1
        return torch.cumsum(tensor * self.scale, dim=1)


def factory(**values) -> nn.Module:
    return FakeState(**values)


def config(**changes) -> SNVBertMambaConfig:
    values = {
        "input_dim": 8,
        "hidden_dim": 16,
        "state_dim": 8,
        "state_size": 2,
        "state_expand": 1,
        "state_head_dim": 2,
        "state_chunk": 4,
        "attention_heads": 2,
        "feedforward_dim": 24,
        "decoder_feedforward_dim": 32,
        "exchange_dim": 8,
        "latent_count": 4,
        "diffusion_radius": 2,
        "diffusion_basis": 2,
        "distance_buckets": 8,
        "distance_limit_bp": 1000,
        "backend": "test",
        "backend_identity": "unit-test",
    }
    values.update(changes)
    return SNVBertMambaConfig(**values)


def batch(pairs: int = 2, loci: int = 7) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    rows = 2 * pairs
    ref_pair = torch.randn(pairs, loci, 8, generator=generator)
    alt_pair = ref_pair + 0.2 * torch.randn(pairs, loci, 8, generator=generator)
    positions_pair = torch.arange(loci).mul(31).add(1).expand(pairs, -1).clone()
    valid_pair = torch.ones(pairs, loci, dtype=torch.bool)
    valid_pair[:, -1] = False
    hidden_pair = torch.zeros(pairs, loci, dtype=torch.bool)
    hidden_pair[:, 2] = True
    genotypes = torch.randint(0, 2, (rows, loci), generator=generator)
    hidden = hidden_pair.repeat_interleave(2, dim=0)
    labels = torch.full_like(genotypes, -100)
    labels[hidden] = genotypes[hidden]
    inputs = {
        "ref_embeddings": ref_pair.repeat_interleave(2, dim=0),
        "alt_embeddings": alt_pair.repeat_interleave(2, dim=0),
        "genotypes": genotypes,
        "positions": positions_pair.repeat_interleave(2, dim=0),
        "chromosomes": torch.full((rows, loci), 21, dtype=torch.long),
        "valid_mask": valid_pair.repeat_interleave(2, dim=0),
        "hidden_mask": hidden,
        "prediction_mask": hidden.clone(),
        "segment_ids": torch.ones(rows, loci, dtype=torch.long),
    }
    return inputs, labels


def swap(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, 2, *tensor.shape[1:])[:, [1, 0]].reshape_as(tensor)


def test_architecture_inventory_and_shape() -> None:
    model = SNVBertMamba(config(), factory)
    FakeState.calls = 0
    inputs, _ = batch()
    output = model(**inputs)
    assert output["logits"].shape == (4, 7, 2)
    assert output["phased_genotype_probability"].shape == (2, 7, 4)
    assert output["architecture"]["pattern"] == ["S", "S", "A"] * 4
    assert FakeState.calls == 16
    assert len(model.backbone.layers) == 12


def test_default_release_is_h0_without_geometry_extension() -> None:
    resolved = config()
    assert resolved.geometry_mode == "none"
    assert resolved.distance_bias is False


def test_external_benchmark_port_is_reserved_without_implementation() -> None:
    request = BenchmarkRequest(
        batch={},
        split="val",
        missing_rate=0.95,
        mask_seed=42,
        selection_identity="unit-selection",
    )
    assert request.split == "val"
    assert ExternalBenchmarkPort.__subclasses__() == []


def test_phase_swap_and_target_hiding() -> None:
    model = SNVBertMamba(config(), factory).eval()
    inputs, _ = batch(pairs=1)
    with torch.no_grad():
        original = model(**inputs)
    swapped_inputs = {key: swap(value) for key, value in inputs.items()}
    with torch.no_grad():
        swapped = model(**swapped_inputs)
    assert torch.allclose(swapped["logits"], swap(original["logits"]), atol=3.0e-6, rtol=3.0e-6)
    assert torch.allclose(swapped["dosage"], original["dosage"], atol=3.0e-6, rtol=3.0e-6)
    assert torch.allclose(swapped["phased_genotype_probability"], original["phased_genotype_probability"][..., [0, 2, 1, 3]], atol=3.0e-6, rtol=3.0e-6)
    changed = {key: value.clone() for key, value in inputs.items()}
    target = changed["prediction_mask"]
    changed["genotypes"][target] = 1 - changed["genotypes"][target]
    with torch.no_grad():
        hidden_changed = model(**changed)
    assert torch.equal(original["logits"], hidden_changed["logits"])


def test_every_parameter_is_connected_to_training_loss() -> None:
    model = SNVBertMamba(config(), factory)
    inputs, labels = batch()
    output = model(**inputs)
    sums, counts = counted_loss_parts(output, labels)
    loss = normalized_loss(sums, counts, 1)
    loss.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    nonfinite = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    assert missing == []
    assert nonfinite == []


def test_optimizer_groups_cover_each_parameter_once() -> None:
    model = SNVBertMamba(config(), factory)
    groups, audit = parameter_groups(model, 0.01)
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    assert len(grouped) == len(set(grouped)) == len(expected)
    assert set(grouped) == set(expected)
    assert audit["all_parameters_classified_once"] is True
    policies = {
        record["name"]: record["policy"] for record in audit["classifications"]
    }
    assert policies["diffusion.kernel"] == "zero_decay"
    assert policies["coordinate.megabase.weight"] == "zero_decay"
    assert policies["backbone.layers.2.attention.query_weight"] == "zero_decay"
    assert policies["backbone.layers.2.exchange.gate.weight"] == "decay"


def test_global_stream_is_unique_and_tail_padded() -> None:
    stream = GlobalWindowStream(282, 128, 8, 0)
    permutation = stream.permutation()
    assert permutation.unique().numel() == 282
    assert stream.updates == 3
    ranks = [list(GlobalWindowStream(282, 128, 8, rank)) for rank in range(8)]
    tail = [ranks[rank][-16:] for rank in range(8)]
    flattened = [value for values in tail for value in values]
    assert sum(value >= 0 for value in flattened) == 26
    assert sum(value < 0 for value in flattened) == 102


def test_source_tree_has_no_retired_namespace_or_external_reference() -> None:
    root = Path(__file__).parents[1] / "src" / "snvbert_mamba"
    retired = "omega" + "_v3"
    external = "var" + "former"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").casefold()
        assert retired not in text
        assert external not in text
