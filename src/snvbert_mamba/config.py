from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SNVBertMambaConfig:
    input_dim: int = 512
    hidden_dim: int = 1024
    state_dim: int = 512
    state_size: int = 64
    state_expand: int = 2
    state_head_dim: int = 64
    state_chunk: int = 256
    state_conv: int = 4
    stages: int = 4
    stage_pattern: str = "S,S,A"
    attention_heads: int = 16
    feedforward_dim: int = 4480
    decoder_feedforward_dim: int = 4096
    local_kernel: int = 5
    exchange_dim: int = 128
    latent_count: int = 64
    diffusion_radius: int = 8
    diffusion_basis: int = 8
    diffusion_max_log_distance: float = 14.0
    dropout: float = 0.0
    norm_epsilon: float = 1.0e-6
    layer_norm_epsilon: float = 1.0e-12
    rope_base: float = 1_000_000.0
    rope_scale: float = 1000.0
    megabase_bins: int = 300
    kilobase_bins: int = 1000
    base_bins: int = 1000
    chromosome_bins: int = 26
    hidden_genotype_blend: float = 0.5
    window_feature_dim: int = 2
    distance_buckets: int = 32
    distance_limit_bp: int = 1_000_000
    geometry_mode: str = "none"
    distance_bias: bool = False
    backend: str = "metax_mamba2"
    backend_identity: str = "unregistered"

    def __post_init__(self) -> None:
        if self.stages != 4 or self.stage_pattern.replace(" ", "").upper() != "S,S,A":
            raise ValueError("stage layout must be [S,S,A] repeated four times")
        if self.local_kernel != 5 or self.dropout != 0.0:
            raise ValueError("local kernel must be 5 and dropout must be zero")
        if self.geometry_mode not in {"none", "film"}:
            raise ValueError("geometry mode must be none or film")
        if self.backend not in {"metax_mamba2", "official_mamba3", "test"}:
            raise ValueError("unsupported state backend")
        if self.backend_identity == "unregistered":
            raise ValueError("backend identity must be explicit")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden dimension must divide attention heads")
        if (self.state_dim * self.state_expand) % self.state_head_dim:
            raise ValueError("expanded state dimension must divide state head dimension")
        values = (
            self.input_dim,
            self.hidden_dim,
            self.state_dim,
            self.state_size,
            self.state_expand,
            self.state_head_dim,
            self.state_chunk,
            self.feedforward_dim,
            self.decoder_feedforward_dim,
            self.exchange_dim,
            self.latent_count,
            self.diffusion_radius,
            self.diffusion_basis,
            self.megabase_bins,
            self.kilobase_bins,
            self.base_bins,
            self.chromosome_bins,
            self.window_feature_dim,
        )
        if min(values) <= 0:
            raise ValueError("all architecture dimensions must be positive")
        if not 0.0 <= float(self.hidden_genotype_blend) <= 1.0:
            raise ValueError("hidden genotype blend must be in [0,1]")
        if float(self.diffusion_max_log_distance) <= 0.0:
            raise ValueError("diffusion distance scale must be positive")
