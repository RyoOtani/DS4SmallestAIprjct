"""
hierarchical_moe.py — Hierarchical Mixture of Experts for Small Models.

Architecture:
  Level 1 (Domain Router): selects among K sub-domains
    ├── General MoE Group (code, text, reasoning)
    │   ├── Expert 0 (FFN, ~50M params)
    │   ├── Expert 1
    │   └── ...
    ├── Code MoE Group
    │   ├── Expert 0 (Python/JS/Systems)
    │   └── ...
    ├── Japanese MoE Group
    │   └── ...
    └── Math/Logic MoE Group
        └── ...

Design goals:
  - Each expert < 0.1B (target: 50-90M params)
  - Top-2 gating at both levels
  - Load balancing loss for uniform expert utilization
  - Shared attention across experts (MoE only in FFN layers)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════

@dataclass
class HierarchicalMoEConfig:
    """Configuration for Hierarchical MoE Small Model."""
    # Model dimensions
    hidden_dim: int = 576
    n_layers: int = 14
    n_heads: int = 9
    head_dim: int = 64
    vocab_size: int = 72000
    max_seq_len: int = 4096

    # Hierarchical MoE
    n_moe_layers: int = 8          # Every other layer is MoE
    n_domain_groups: int = 4       # general, code, japanese, math
    n_experts_per_group: int = 6   # experts per domain group
    n_active_experts: int = 2      # top-2 gating per group
    expert_ffn_dim: int = 1536     # FFN intermediate dim per expert

    # Expert param count ~ hidden_dim * expert_ffn_dim * 3 ≈ 576*1536*3 ≈ 2.6M (tiny!)
    # Actually: gate(576*1536) + up(576*1536) + down(1536*576) = 3*576*1536 ≈ 2.65M
    # 6 experts * 4 groups = 24 experts * 2.65M ≈ 63.7M total expert params
    # With shared attention and embeddings, total < 150M → well under 0.1B per expert ✅

    # Training
    load_balance_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    dropout: float = 0.1

    # Domain labels (for domain router training)
    domain_labels: List[str] = field(default_factory=lambda: [
        "general", "code", "japanese", "math"
    ])


# ═══════════════════════════════════════════════════════
# Expert FFN (lightweight, <0.1B each)
# ═══════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    """A single expert FFN with SwiGLU activation."""
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.dropout(self.down(
            F.silu(self.gate(x)) * self.up(x)
        ))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════
# Domain Router (Level 1)
# ═══════════════════════════════════════════════════════

class DomainRouter(nn.Module):
    """Routes tokens to domain groups based on hidden state."""
    def __init__(self, hidden_dim: int, n_domains: int, top_k: int = 2):
        super().__init__()
        self.n_domains = n_domains
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, n_domains, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, hidden_dim]
        Returns:
            weights: [batch, seq, top_k] — routing weights
            indices: [batch, seq, top_k] — selected domain indices
            logits:  [batch, seq, n_domains] — raw router logits
        """
        logits = self.router(x)  # [B, S, n_domains]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices, logits


# ═══════════════════════════════════════════════════════
# Expert Router (Level 2, per domain group)
# ═══════════════════════════════════════════════════════

class ExpertRouter(nn.Module):
    """Routes tokens to experts within a domain group."""
    def __init__(self, hidden_dim: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)  # [B, S, n_experts]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices, logits


# ═══════════════════════════════════════════════════════
# Hierarchical MoE Layer
# ═══════════════════════════════════════════════════════

class HierarchicalMoELayer(nn.Module):
    """
    One transformer layer with Hierarchical MoE FFN.

    Flow:
      hidden → RMSNorm → Attention (+ residual)
             → RMSNorm → DomainRouter → ExpertRouter → ExpertFFN (+ residual)
    """
    def __init__(self, config: HierarchicalMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        D = config.hidden_dim

        # Attention (shared — not expert)
        self.attn_norm = nn.RMSNorm(D)
        self.attn = nn.MultiheadAttention(
            D, config.n_heads, dropout=config.dropout,
            batch_first=True
        )

        # MoE FFN
        self.ffn_norm = nn.RMSNorm(D)

        # Level 1: Domain Router
        self.domain_router = DomainRouter(
            D, config.n_domain_groups, top_k=config.n_active_experts
        )

        # Level 2: Expert Routers (one per domain group)
        self.expert_routers = nn.ModuleList([
            ExpertRouter(D, config.n_experts_per_group, top_k=config.n_active_experts)
            for _ in range(config.n_domain_groups)
        ])

        # Experts: [n_domains × n_experts] FFNs
        self.experts = nn.ModuleList([
            nn.ModuleList([
                ExpertFFN(D, config.expert_ffn_dim, config.dropout)
                for _ in range(config.n_experts_per_group)
            ])
            for _ in range(config.n_domain_groups)
        ])

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        B, S, D = x.shape

        # ── Attention ──
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = residual + attn_out

        # ── Hierarchical MoE FFN ──
        residual = x
        x_normed = self.ffn_norm(x)
        ffn_out = torch.zeros_like(x_normed)
        aux_loss = 0.0

        # Level 1: Route to domain groups
        domain_weights, domain_indices, domain_logits = self.domain_router(x_normed)
        # domain_weights: [B, S, top_k], domain_indices: [B, S, top_k]

        # Level 2: Within each domain group, route to experts
        for g in range(self.config.n_domain_groups):
            # Find tokens routed to this domain group
            mask_g = (domain_indices == g).any(dim=-1)  # [B, S]
            if not mask_g.any():
                continue

            # Get tokens for this group
            x_g = x_normed[mask_g]  # [N_g, D]

            # Route to experts within group
            expert_weights, expert_indices, expert_logits = \
                self.expert_routers[g](x_g)
            # expert_weights: [N_g, top_k], expert_indices: [N_g, top_k]

            # Compute expert outputs
            group_out = torch.zeros_like(x_g)
            for k in range(self.config.n_active_experts):
                e_idx = expert_indices[:, k]  # [N_g]
                e_weight = expert_weights[:, k:k+1]  # [N_g, 1]

                # Process per-expert (batched would be faster but complex)
                for e in range(self.config.n_experts_per_group):
                    token_mask = (e_idx == e)
                    if not token_mask.any():
                        continue
                    expert_out = self.experts[g][e](x_g[token_mask])
                    group_out[token_mask] += e_weight[token_mask] * expert_out

            # Place back into full output
            ffn_out_flat = ffn_out.view(-1, D)
            mask_flat = mask_g.view(-1)
            ffn_out_flat[mask_flat] = group_out
            ffn_out = ffn_out_flat.view(B, S, D)

            # Aux loss: load balancing
            aux_loss += self._load_balance_loss(
                domain_logits, expert_logits, g, mask_g
            )

        x = residual + ffn_out

        stats = {"aux_loss": aux_loss, "layer": self.layer_idx}
        return x, stats

    def _load_balance_loss(self, domain_logits: torch.Tensor,
                           expert_logits: torch.Tensor,
                           group_idx: int, mask: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for expert utilization."""
        if not mask.any():
            return torch.tensor(0.0, device=domain_logits.device)

        # Fraction of tokens dispatched to each expert
        expert_probs = F.softmax(expert_logits, dim=-1)  # [N_g, n_experts]
        mean_probs = expert_probs.mean(dim=0)  # [n_experts]
        n_experts = expert_probs.shape[-1]
        target = torch.ones(n_experts, device=expert_logits.device) / n_experts
        loss = F.kl_div(mean_probs.log(), target, reduction='batchmean')
        return loss * self.config.load_balance_coef

    @property
    def expert_param_counts(self) -> List[int]:
        return [
            sum(e.param_count for e in group_experts)
            for group_experts in self.experts
        ]


# ═══════════════════════════════════════════════════════
# Full Hierarchical MoE Model
# ═══════════════════════════════════════════════════════

class HierarchicalMoEModel(nn.Module):
    """Complete Small Model with Hierarchical MoE architecture."""

    def __init__(self, config: HierarchicalMoEConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer layers (MoE on even layers, dense on odd)
        self.layers = nn.ModuleList([
            HierarchicalMoELayer(config, i)
            if i % 2 == 0 and i < config.n_moe_layers * 2
            else self._make_dense_layer(config)
            for i in range(config.n_layers)
        ])

        # Output
        self.final_norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.tok_embeddings.weight

        self._print_model_info()

    def _make_dense_layer(self, config: HierarchicalMoEConfig) -> nn.Module:
        """Create a standard dense transformer layer for non-MoE positions."""
        return nn.ModuleDict({
            "attn_norm": nn.RMSNorm(config.hidden_dim),
            "attn": nn.MultiheadAttention(
                config.hidden_dim, config.n_heads,
                dropout=config.dropout, batch_first=True
            ),
            "ffn_norm": nn.RMSNorm(config.hidden_dim),
            "ffn": nn.Sequential(
                nn.Linear(config.hidden_dim, config.expert_ffn_dim * 2, bias=False),
                nn.SiLU(),
                nn.Linear(config.expert_ffn_dim * 2, config.hidden_dim, bias=False),
                nn.Dropout(config.dropout),
            ),
        })

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        B, S = input_ids.shape
        x = self.tok_embeddings(input_ids)

        total_aux_loss = 0.0
        stats = {}

        for i, layer in enumerate(self.layers):
            if isinstance(layer, HierarchicalMoELayer):
                x, layer_stats = layer(x, attention_mask)
                total_aux_loss += layer_stats.get("aux_loss", 0.0)
            else:
                # Dense layer
                residual = x
                x = layer["attn_norm"](x)
                attn_out, _ = layer["attn"](x, x, x, attn_mask=attention_mask)
                x = residual + attn_out
                residual = x
                x = layer["ffn_norm"](x)
                x = residual + layer["ffn"](x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        stats["aux_loss"] = total_aux_loss
        return logits, stats

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        expert_total = sum(
            sum(e.param_count for e in group)
            for layer in self.layers
            if isinstance(layer, HierarchicalMoELayer)
            for group in layer.experts
        )
        per_expert = expert_total / (
            self.config.n_moe_layers *
            self.config.n_domain_groups *
            self.config.n_experts_per_group
        ) if self.config.n_moe_layers > 0 else 0

        print(f"🏗️  Hierarchical MoE Small Model")
        print(f"   Total params: {total/1e6:.1f}M")
        print(f"   Expert params: {expert_total/1e6:.1f}M")
        print(f"   Per-expert: {per_expert/1e6:.2f}M (<0.1B ✅)" if per_expert < 100e6
              else f"   Per-expert: {per_expert/1e6:.1f}M (⚠️  >0.1B)")
        print(f"   Architecture: {self.config.n_layers}L × {self.config.hidden_dim}D")
        print(f"   MoE: {self.config.n_moe_layers} layers, "
              f"{self.config.n_domain_groups} domains × "
              f"{self.config.n_experts_per_group} experts")


# ═══════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════

def create_small_model(vocab_size: int = 72000) -> HierarchicalMoEModel:
    """Create the default Small Model with Hierarchical MoE."""
    config = HierarchicalMoEConfig(
        hidden_dim=576,
        n_layers=14,
        n_heads=9,
        head_dim=64,
        vocab_size=vocab_size,
        max_seq_len=4096,
        n_moe_layers=8,
        n_domain_groups=4,
        n_experts_per_group=6,
        n_active_experts=2,
        expert_ffn_dim=1536,
    )
    return HierarchicalMoEModel(config)


# ═══════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    model = create_small_model(vocab_size=72000)

    # Test forward pass
    dummy = torch.randint(0, 72000, (2, 128))
    with torch.no_grad():
        logits, stats = model(dummy)
    print(f"\n✅ Forward pass: {logits.shape} (expected [2, 128, 72000])")
    print(f"   Aux loss: {stats['aux_loss']:.6f}")
    print(f"   Memory: {logits.element_size() * logits.numel() / 1e6:.1f} MB")

    # Per-expert check
    for i, layer in enumerate(model.layers):
        if isinstance(layer, HierarchicalMoELayer):
            counts = layer.expert_param_counts
            for g, count in enumerate(counts):
                per_exp = count / model.config.n_experts_per_group
                print(f"   Layer {i} Group {g}: {count/1e6:.1f}M total, "
                      f"{per_exp/1e6:.2f}M/expert {'✅' if per_exp < 100e6 else '⚠️'}")
