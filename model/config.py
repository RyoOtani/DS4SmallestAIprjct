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
        """Rough estimate of total parameters."""
        D = self.hidden_dim
        L = self.n_layers
        V = self.vocab_size

        # Embedding: V * D
        emb = V * D

        # Per layer:
        # Attention: 4 * D * D (Q, K, V, O) — simplified
        attn = 4 * D * D
        # MoE: n_experts * 3 * D * inter_dim
        moe = self.n_experts * 3 * D * self.expert_inter_dim
        # FFN: 3 * D * ffn_inter_dim
        ffn = 3 * D * self.ffn_inter_dim
        # RMSNorm: 2 * D
        norm = 2 * D

        per_layer = attn + moe + ffn + norm
        total = emb + L * per_layer

        # LM head (if not tied)
        if not self.tie_word_embeddings:
            total += V * D

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
}


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
