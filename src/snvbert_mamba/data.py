from __future__ import annotations

import json
import bisect
from pathlib import Path

import torch
from torch.utils.data import Dataset


REQUIRED = (
    "ref_embeddings",
    "alt_embeddings",
    "genotypes",
    "positions",
    "chromosomes",
    "valid_mask",
    "hidden_mask",
    "prediction_mask",
    "labels",
)


class TensorWindowDataset(Dataset):
    def __init__(self, manifest: str | Path) -> None:
        path = Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.root = path.parent
        self.files = tuple(str(value) for value in payload["files"])
        if not self.files:
            raise ValueError("manifest contains no windows")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dummy = int(index) < 0
        source = 0 if dummy else int(index)
        item = torch.load(self.root / self.files[source], map_location="cpu", weights_only=True)
        missing = [key for key in REQUIRED if key not in item]
        if missing:
            raise RuntimeError(f"window is missing tensors: {missing}")
        result = {key: value.clone() for key, value in item.items() if torch.is_tensor(value)}
        rows = result["labels"].shape[0]
        result["sample_weight"] = torch.full((rows,), 0.0 if dummy else 1.0)
        if dummy:
            result["labels"].fill_(-100)
            result["hidden_mask"].zero_()
            result["prediction_mask"].zero_()
        return result


def collate_windows(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise RuntimeError("window tensor schemas differ")
    return {key: torch.cat([item[key] for item in items], dim=0) for key in sorted(keys)}


def _tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")


class PackedWindowDataset(Dataset):
    def __init__(self, root: str | Path, maximum_length: int = 1024, mask_rate: float = 0.4, seed: int = 42, indices_name: str = "stratified_indices.pt") -> None:
        self.root = Path(root)
        self.maximum_length = int(maximum_length)
        self.mask_rate = float(mask_rate)
        self.seed = int(seed)
        self.epoch = 0
        self.packs = sorted(path for path in self.root.glob("pack_*") if (path / "genotypes.zarr").exists())
        if not self.packs:
            raise FileNotFoundError("no genotype packs were found")
        self.metadata = []
        self.indices = []
        self.cumulative = []
        self.feature_cache = {}
        self.matrix_cache = {}
        total = 0
        for pack in self.packs:
            meta = json.loads((pack / "dataset_metadata.json").read_text(encoding="utf-8"))
            segment_path = pack / "sequence_segments.json"
            segments = json.loads(segment_path.read_text(encoding="utf-8")) if segment_path.exists() else [{"start": 0, "end": int(meta["num_variants"]), "feature_source_dir": "."}]
            index = _tensor(pack / indices_name).long()
            if index.dim() != 2 or index.shape[1] != 3:
                raise ValueError("window indices must have shape [N,3]")
            self.metadata.append({"pack": pack, "segments": segments, "input_dim": int(meta["input_feature_dim"])})
            self.indices.append(index)
            total += index.shape[0]
            self.cumulative.append(total)

    @property
    def input_dim(self) -> int:
        return int(self.metadata[0]["input_dim"])

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _matrix(self, pack_index: int):
        if pack_index not in self.matrix_cache:
            import zarr

            self.matrix_cache[pack_index] = zarr.open(str(self.packs[pack_index] / "genotypes.zarr"), mode="r")["matrix"]
        return self.matrix_cache[pack_index]

    def _segment(self, pack_index: int, start: int, end: int):
        pack = self.packs[pack_index]
        for segment in self.metadata[pack_index]["segments"]:
            segment_start = int(segment.get("start", 0))
            segment_end = int(segment.get("end", 0))
            if start >= segment_start and end <= segment_end:
                source = Path(str(segment["feature_source_dir"]))
                if not source.is_absolute():
                    source = (pack / source).resolve()
                offset = int(segment.get("global_site_offset", segment_start))
                return start - segment_start, end - segment_start, source, offset
        raise ValueError("window crosses an undeclared segment")

    def _features(self, source: Path) -> dict[str, torch.Tensor]:
        if source not in self.feature_cache:
            positions = _tensor(source / "positions.pt").long()
            chromosome_path = source / "chrom_indices.pt"
            chromosomes = _tensor(chromosome_path).long() if chromosome_path.exists() else torch.full_like(positions, 21)
            self.feature_cache[source] = {
                "ref": _tensor(source / "ref_matrix.pt").float(),
                "alt": _tensor(source / "alt_matrix.pt").float(),
                "positions": positions,
                "chromosomes": chromosomes,
            }
        return self.feature_cache[source]

    def _mask(self, valid: torch.Tensor, index: int) -> torch.Tensor:
        valid_sites = valid.all(dim=0).bool()
        generator = torch.Generator().manual_seed(
            self.seed + int(index) + 1_000_003 * self.epoch
        )
        return (
            torch.rand(tuple(valid_sites.shape), generator=generator)
            < self.mask_rate
        ) & valid_sites

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        dummy = int(index) < 0
        source_index = 0 if dummy else int(index)
        if source_index >= len(self):
            raise IndexError(source_index)
        pack_index = bisect.bisect_right(self.cumulative, source_index)
        previous = 0 if pack_index == 0 else self.cumulative[pack_index - 1]
        anchor, start, end = (int(value) for value in self.indices[pack_index][source_index - previous].tolist())
        length = end - start
        if length <= 0 or length > self.maximum_length:
            raise ValueError("invalid window length")
        local_start, local_end, feature_source, global_offset = self._segment(pack_index, start, end)
        genotype_array = self._matrix(pack_index).get_orthogonal_selection(([anchor, anchor + 1], slice(start, end)))
        genotypes = torch.full((2, self.maximum_length), -100, dtype=torch.long)
        genotypes[:, :length] = torch.as_tensor(genotype_array, dtype=torch.long)
        valid = genotypes.eq(0) | genotypes.eq(1)
        features = self._features(feature_source)
        ref = torch.zeros((2, self.maximum_length, self.input_dim), dtype=torch.float32)
        alt = torch.zeros_like(ref)
        positions = torch.zeros((2, self.maximum_length), dtype=torch.long)
        chromosomes = torch.zeros_like(positions)
        ref_block = features["ref"][local_start:local_end]
        alt_block = features["alt"][local_start:local_end]
        position_block = features["positions"][local_start:local_end]
        chromosome_block = features["chromosomes"][local_start:local_end]
        ref[:, :length] = ref_block.unsqueeze(0).expand(2, -1, -1)
        alt[:, :length] = alt_block.unsqueeze(0).expand(2, -1, -1)
        positions[:, :length] = position_block.unsqueeze(0).expand(2, -1)
        chromosomes[:, :length] = chromosome_block.unsqueeze(0).expand(2, -1)
        locus_mask = self._mask(valid, source_index)
        hidden = locus_mask.unsqueeze(0).expand(2, -1).clone()
        labels = torch.full_like(genotypes, -100)
        labels[hidden & valid] = genotypes[hidden & valid]
        weight = torch.ones(2, dtype=torch.float32)
        if dummy:
            hidden.zero_()
            labels.fill_(-100)
            weight.zero_()
        return {
            "ref_embeddings": ref,
            "alt_embeddings": alt,
            "genotypes": genotypes,
            "positions": positions,
            "chromosomes": chromosomes,
            "valid_mask": valid,
            "hidden_mask": hidden,
            "prediction_mask": hidden.clone(),
            "segment_ids": chromosomes.clone(),
            "labels": labels,
            "sample_weight": weight,
        }
