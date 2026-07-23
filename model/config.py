"""TinyLLM Model Configurations — Multi-scale model definitions."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Complete model configuration covering all architectural decisions."""

    # ── Basic dimensions ─────────────────────────────────────────────
    name: str = "tinyllm-nano"
    hidden_dim: int = 2048
    vocab_size: int = 65536
    max_seq_len: int = 8192
    n_layers: int = 32

    # ── Attention ────────────────────────────────────────────────────
    n_heads: int = 32
    n_kv_heads: int = 8        # GQA: fewer KV heads
    head_dim: int = 128
    use_mla: bool = True       # Multi-head Latent Attention
    kv_latent_dim: int = 512   # compressed KV latent
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None  # {"type": "linear", "factor": 2.0}
    sliding_window: int = 0    # 0 = disabled
    attn_dropout: float = 0.0

    # ── MoE ──────────────────────────────────────────────────────────
    use_moe: bool = True
    n_experts: int = 64
    n_active_experts: int = 6  # top-k
    expert_inter_dim: int = 1024
    moe_layers: list[int] = field(default_factory=lambda: [])  # [] = all layers
    moe_aux_loss_weight: float = 0.01
    expert_capacity_factor: float = 1.25
    shared_experts: int = 2    # shared across all tokens

    # ── FFN ──────────────────────────────────────────────────────────
    ffn_inter_dim: int = 5632
    ffn_activation: str = "swiglu"
    ffn_multiple_of: int = 256  # inter_dim must be multiple of this

    # ── Normalization ────────────────────────────────────────────────
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5

    # ── Multi-Token Prediction (MTP) ─────────────────────────────────
    use_mtp: bool = True
    mtp_depth: int = 1         # predict next N tokens (additional heads)
    mtp_share_embedding: bool = False

    # ── Embeddings ───────────────────────────────────────────────────
    tie_word_embeddings: bool = False
    embedding_multiplier: float = 1.0  # scale embeddings (Gemma style)

    # ── Quantization (for training) ──────────────────────────────────
    train_dtype: str = "bfloat16"  # float32, float16, bfloat16
    use_fp8_training: bool = False
    use_qk_norm: bool = False

    # ── Regularization ───────────────────────────────────────────────
    dropout: float = 0.0
    weight_decay: float = 0.1
    gradient_clip: float = 1.0

    # ── Initialization ───────────────────────────────────────────────
    init_std: float = 0.02
    init_type: str = "normal"  # normal, xavier_uniform, kaiming

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer_type: str = "bpe"
    tokenizer_vocab_size: int = 65536

    @property
    def n_kv_heads_effective(self) -> int:
        """Effective KV heads (GQA)."""
        return self.n_kv_heads if self.n_kv_heads > 0 else self.n_heads

    @property
    def total_params_estimate(self) -> int:
        """Accurate total parameter count."""
        D = self.hidden_dim
        L = self.n_layers
        V = self.vocab_size
        H = self.n_heads
        d = self.head_dim

        # Embedding
        emb = V * D

        per_layer = 0

        # Attention: q_proj + kv_compress + k_up + v_up + o_proj
        if self.use_mla:
            attn = (H * d * D) + (self.kv_latent_dim * D) + (H * d * self.kv_latent_dim) * 2 + (H * d * D)
        else:
            attn = (H * d * D) + (self.n_kv_heads_effective * d * D) * 2 + (H * d * D)
        per_layer += attn

        # RMS Norm × 2
        per_layer += 2 * D

        # MoE: gate (n_experts * D) + n_experts × (3 × inter_dim × D) + shared × (3 × inter_dim × D)
        if self.use_moe:
            gate_params = self.n_experts * D
            expert_params = self.n_experts * 3 * D * self.expert_inter_dim
            shared_params = self.shared_experts * 3 * D * self.expert_inter_dim
            per_layer += gate_params + expert_params + shared_params
        else:
            per_layer += 3 * D * self.ffn_inter_dim  # SwiGLU: gate + up + down

        total = emb + L * per_layer

        # LM head (if not tied)
        if not self.tie_word_embeddings:
            total += V * D

        # MTP heads: depth × (2*D^2 + 4*D^2 * depth_sub + D*V)
        if self.use_mtp and self.mtp_depth > 0:
            mtp_params = self.mtp_depth * (
                2 * D * D +  # input_proj
                2 * 3 * D * (D * 4) +  # sub-layer FFNs (2 sub-layers)
                D +  # final_norm
                D * V  # output projection
            )
            total += mtp_params

        return total

    @property
    def active_params_estimate(self) -> int:
        """Rough estimate of active parameters (non-expert params + top-k experts)."""
        D = self.hidden_dim
        L = self.n_layers
        V = self.vocab_size
        k = self.n_active_experts
        N = self.n_experts

        emb = V * D
        attn = 4 * D * D
        # Active MoE: only k experts active
        moe_active = k * 3 * D * self.expert_inter_dim
        ffn = 3 * D * self.ffn_inter_dim
        norm = 2 * D

        per_layer = attn + moe_active + ffn + norm
        total = emb + L * per_layer
        return total


# ═══════════════════════════════════════════════════════════════════════
# Pre-defined model configurations at multiple scales
# ═══════════════════════════════════════════════════════════════════════

TINYLLM_CONFIGS = {
    # ── Nano: 600M active / 2B total ─────────────────────────────
    "nano": ModelConfig(
        name="tinyllm-nano",
        hidden_dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
        use_mla=True,
        kv_latent_dim=256,
        ffn_inter_dim=2816,
        use_moe=True,
        n_experts=32,
        n_active_experts=4,
        expert_inter_dim=512,
        shared_experts=1,
        use_mtp=True,
        mtp_depth=1,
    ),

    # ── Small: 2.4B active / 10B total ──────────────────────────
    "small": ModelConfig(
        name="tinyllm-small",
        hidden_dim=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=512,
        ffn_inter_dim=5632,
        use_moe=True,
        n_experts=64,
        n_active_experts=6,
        expert_inter_dim=1024,
        shared_experts=2,
        use_mtp=True,
        mtp_depth=2,
    ),

    # ── Medium: 7B active / 30B total ───────────────────────────
    "medium": ModelConfig(
        name="tinyllm-medium",
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=512,
        ffn_inter_dim=14336,
        use_moe=True,
        n_experts=128,
        n_active_experts=8,
        expert_inter_dim=2048,
        shared_experts=2,
        use_mtp=True,
        mtp_depth=4,
        sliding_window=4096,
    ),

    # ── Large: 20B active / 80B total ───────────────────────────
    "large": ModelConfig(
        name="tinyllm-large",
        hidden_dim=6144,
        n_layers=48,
        n_heads=48,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=768,
        ffn_inter_dim=18432,
        use_moe=True,
        n_experts=256,
        n_active_experts=8,
        expert_inter_dim=3072,
        shared_experts=4,
        use_mtp=True,
        mtp_depth=4,
        sliding_window=4096,
        rope_scaling={"type": "linear", "factor": 2.0},
        use_qk_norm=True,
    ),

    # ── Dense (no MoE): ~7B ─────────────────────────────────────
    "dense-7b": ModelConfig(
        name="tinyllm-dense-7b",
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        head_dim=128,
        use_mla=False,
        ffn_inter_dim=14336,
        use_moe=False,
        use_mtp=True,
        mtp_depth=2,
    ),

    # ═══════════════════════════════════════════════════════════════
    # SUPER-SCALE MODELS (200B → 2.5T total params)
    # ═══════════════════════════════════════════════════════════════

    # ── X-Large: 40B active / 200B total ─────────────────────────
    "xlarge": ModelConfig(
        name="tinyllm-xlarge",
        hidden_dim=6144,
        n_layers=48,
        n_heads=48,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=768,
        ffn_inter_dim=16384,
        use_moe=True,
        n_experts=128,
        n_active_experts=8,
        expert_inter_dim=3072,
        shared_experts=4,
        use_mtp=True,
        mtp_depth=4,
        sliding_window=4096,
        rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192},
        use_qk_norm=True,
        train_dtype="bfloat16",
        gradient_clip=0.5,
    ),

    # ── XX-Large: 90B active / 500B total ────────────────────────
    "xxlarge": ModelConfig(
        name="tinyllm-xxlarge",
        hidden_dim=8192,
        n_layers=64,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=1024,
        ffn_inter_dim=20480,
        use_moe=True,
        n_experts=192,
        n_active_experts=8,
        expert_inter_dim=4096,
        shared_experts=6,
        use_mtp=True,
        mtp_depth=6,
        sliding_window=4096,
        rope_scaling={"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192},
        use_qk_norm=True,
        train_dtype="bfloat16",
        gradient_clip=0.3,
    ),

    # ── Mega: 180B active / 1T total ─────────────────────────────
    "mega": ModelConfig(
        name="tinyllm-mega",
        hidden_dim=10240,
        n_layers=72,
        n_heads=80,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=1280,
        ffn_inter_dim=25600,
        use_moe=True,
        n_experts=256,
        n_active_experts=8,
        expert_inter_dim=5120,
        shared_experts=8,
        use_mtp=True,
        mtp_depth=8,
        sliding_window=4096,
        rope_scaling={"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 8192},
        use_qk_norm=True,
        train_dtype="bfloat16",
        gradient_clip=0.2,
        max_seq_len=16384,
        vocab_size=131072,
    ),

    # ── Giga: 350B active / 2T total ────────────────────────────
    "giga": ModelConfig(
        name="tinyllm-giga",
        hidden_dim=12288,
        n_layers=88,
        n_heads=96,
        n_kv_heads=8,
        head_dim=128,
        use_mla=True,
        kv_latent_dim=1536,
        ffn_inter_dim=30720,
        use_moe=True,
        n_experts=320,
        n_active_experts=8,
        expert_inter_dim=6144,
        shared_experts=12,
        use_mtp=True,
        mtp_depth=8,
        sliding_window=8192,
        rope_scaling={"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192},
        use_qk_norm=True,
        train_dtype="bfloat16",
        gradient_clip=0.1,
        max_seq_len=32768,
        vocab_size=131072,
    ),
}


def create_custom_config(
    total_params_b: float,
    active_ratio: float = 0.08,
    n_layers: int = 0,
    name: str = "tinyllm-custom",
) -> ModelConfig:
    """Create a model config targeting a specific total parameter count.

    Args:
        total_params_b: Target total parameters in billions (e.g. 500 for 500B)
        active_ratio: Desired active/total ratio (default 8% via MoE)
        n_layers: Number of layers (0 = auto-select based on size)
        name: Model name

    Returns:
        ModelConfig with dimensions scaled to the target parameter count

    Scaling follows Kaplan et al. optimal ratios:
      - hidden_dim scales with sqrt(total_params)
      - n_layers scales with ~log(total_params)
      - ffn_inter_dim ≈ 3.5 × hidden_dim
    """
    import math

    # Total parameters including embedding (which is V * D)
    # We'll solve for hidden_dim given the asymptotic layer contribution
    total = total_params_b * 1e9

    if n_layers <= 0:
        # Depth scaling: large models need more layers but diminishing returns
        n_layers = max(24, int(12 * math.log(total_params_b + 1)))

    # Estimate: total ≈ n_layers * (4*D^2 + n_experts * 3*D*inter_dim + 3*D*ffn_inter + 2*D) + 2*V*D
    # The dominant term is MoE: n_layers * n_experts * 3*D*expert_inter_dim
    # Let expert_inter_dim ≈ D/2 (optimized for MoE ratio)
    # Then total ≈ n_layers * n_experts * 3 * D * (D/2) = 1.5 * n_layers * n_experts * D^2
    # So D ≈ sqrt(total / (1.5 * n_layers * n_experts))

    n_experts = max(64, int(total_params_b * 1.5))  # ~1.5 experts per B params
    n_experts = (n_experts // 64) * 64  # round to 64
    n_experts = min(n_experts, 2048)

    # Solve for hidden_dim
    denom = 1.5 * n_layers * n_experts
    D = int(math.sqrt(total / max(denom, 1)))
    D = ((D // 256) + 1) * 256  # round to multiple of 256

    # Clamp to reasonable range
    D = max(1024, min(D, 24576))

    # n_heads: 1 head per 128 dim
    n_heads = D // 128
    n_heads = max(8, (n_heads // 8) * 8)  # multiple of 8

    # Expert inter dim = D/2 (for ~1:4 compute ratio)
    expert_inter_dim = ((D // 2) // 256 + 1) * 256

    # FFN inter dim = 3.5 * D
    ffn_inter_dim = ((int(D * 3.5)) // 256 + 1) * 256

    # KV latent dim
    kv_latent_dim = ((D // 8) // 64 + 1) * 64

    n_active = 8 if total_params_b > 100 else 6
    mtp_depth = min(8, max(1, int(math.log2(total_params_b + 1)) - 1))

    return ModelConfig(
        name=name,
        hidden_dim=D,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=min(8, n_heads // 4),
        head_dim=128,
        use_mla=True,
        kv_latent_dim=kv_latent_dim,
        ffn_inter_dim=ffn_inter_dim,
        use_moe=True,
        n_experts=n_experts,
        n_active_experts=n_active,
        expert_inter_dim=expert_inter_dim,
        shared_experts=max(1, n_experts // 64),
        use_mtp=mtp_depth > 0,
        mtp_depth=mtp_depth,
        sliding_window=4096 if D > 4096 else 0,
        use_qk_norm=D > 8192,
        max_seq_len=16384 if D > 8192 else 8192,
        vocab_size=131072 if D > 8192 else 65536,
    )


def get_config(name: str) -> ModelConfig:
    """Get a pre-defined model configuration."""
    if name in TINYLLM_CONFIGS:
        return TINYLLM_CONFIGS[name]
    raise KeyError(f"Unknown config: {name}. Available: {list(TINYLLM_CONFIGS.keys())}")


def list_configs() -> list[dict]:
    """List all available configs with parameter estimates."""
    return [
        {
            "name": cfg.name,
            "hidden_dim": cfg.hidden_dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "moe": f"{cfg.n_experts}x{cfg.n_active_experts}" if cfg.use_moe else "dense",
            "total_params": f"{cfg.total_params_estimate / 1e9:.1f}B",
            "active_params": f"{cfg.active_params_estimate / 1e9:.1f}B",
        }
        for cfg in TINYLLM_CONFIGS.values()
    ]
