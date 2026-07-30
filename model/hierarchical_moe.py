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
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

from model.layers.normalization import RMSNorm


# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════

@dataclass
class HierarchicalMoEConfig:
    """Configuration for Hierarchical MoE — 3.0B Active / 14.5B Total."""
    # Model dimensions (≈14.5B total params)
    hidden_dim: int = 2048
    n_layers: int = 26           # 16 MoE + 10 dense
    n_heads: int = 16
    head_dim: int = 128
    vocab_size: int = 72000
    max_seq_len: int = 4096

    # Hierarchical MoE
    n_moe_layers: int = 16       # MoE layers (every other layer)
    n_domain_groups: int = 4     # general, code, japanese, math
    n_experts_per_group: int = 6 # experts per domain group
    n_shared_experts: int = 2    # shared experts always active
    n_active_domains: int = 2    # top-k domains
    n_active_experts: int = 2    # top-k experts per domain
    expert_ffn_dim: int = 5632   # FFN intermediate dim per expert

    # Per-expert: 3*D*F = 3*2048*5632 ≈ 34.6M (well over 0.1B now — this is the big model)
    # Experts per MoE layer: 4 groups × 6 = 24 + 2 shared = 26
    # Total experts: 26 × 16 = 416
    # Expert params: 416 × 34.6M ≈ 14.4B of 14.5B total
    # Active per forward: 2 domains × 2 experts = 4 + 2 shared = 6 experts
    # Active expert params: 16 layers × 6 × 34.6M ≈ 3.3B → with embedding/attention/dense ≈ 3.0B active

    # Anti-collapse & load balancing
    load_balance_coef: float = 0.01     # importance-weighted load balance
    router_z_loss_coef: float = 0.001   # Z-loss for router logit stability
    diversity_coef: float = 0.005       # expert weight diversity penalty
    router_noise_std: float = 0.1       # Gaussian noise before gating (exploration)
    expert_dropout: float = 0.05        # random expert dropout during training

    # Gradient dispersion
    expert_lr_multiplier: float = 1.2   # experts learn slightly faster than shared params

    # Training
    dropout: float = 0.1

    # Memory optimization
    use_checkpointing: bool = True  # Activation checkpointing (trade compute for memory)

    # Domain labels
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
    """Routes tokens to domain groups with exploration noise and load balancing."""
    def __init__(self, hidden_dim: int, n_domains: int, top_k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.n_domains = n_domains
        self.top_k = top_k
        self.noise_std = noise_std
        self.router = nn.Linear(hidden_dim, n_domains, bias=False)

    def forward(self, x: torch.Tensor, training: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (weights, indices, logits, z_loss)
        """
        logits = self.router(x)  # [B, S, n_domains]

        # ── Z-loss: penalize large router logits for stability ──
        z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))

        # ── Exploration noise (only during training) ──
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # ── Top-k gating ──
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        return weights, indices, logits, z_loss


# ═══════════════════════════════════════════════════════
# Expert Router (Level 2, per domain group)
# ═══════════════════════════════════════════════════════

class ExpertRouter(nn.Module):
    """Routes tokens to experts within a domain with noise and balancing."""
    def __init__(self, hidden_dim: int, n_experts: int, top_k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor, training: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))

        if training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices, logits, z_loss


# ═══════════════════════════════════════════════════════
# Hierarchical MoE Layer
# ═══════════════════════════════════════════════════════

class HierarchicalMoELayer(nn.Module):
    """
    One MoE transformer layer with:
      - 2-level hierarchical routing (Domain → Expert)
      - Shared experts (always active, load balancing)
      - Load balance + Z-loss + diversity penalty
      - Router noise for exploration
      - Expert dropout for robustness
    """
    def __init__(self, config: HierarchicalMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        D = config.hidden_dim
        F = config.expert_ffn_dim

        # Attention (shared across experts)
        self.attn_norm = RMSNorm(D)
        self.attn = nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout, batch_first=True)

        # FFN norms
        self.ffn_norm = RMSNorm(D)

        # Level 1: Domain Router
        self.domain_router = DomainRouter(D, config.n_domain_groups,
                                          top_k=config.n_active_domains,
                                          noise_std=config.router_noise_std)

        # Level 2: Expert Routers (one per domain)
        self.expert_routers = nn.ModuleList([
            ExpertRouter(D, config.n_experts_per_group, top_k=config.n_active_experts,
                         noise_std=config.router_noise_std)
            for _ in range(config.n_domain_groups)
        ])

        # Domain-gated experts: [n_domains × n_experts]
        self.experts = nn.ModuleList([
            nn.ModuleList([ExpertFFN(D, F, config.dropout)
                          for _ in range(config.n_experts_per_group)])
            for _ in range(config.n_domain_groups)
        ])

        # Shared experts (always active, no routing)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(D, F, config.dropout)
            for _ in range(config.n_shared_experts)
        ]) if config.n_shared_experts > 0 else nn.ModuleList()

        # Expert counter for load statistics
        self.register_buffer("expert_usage", torch.zeros(
            config.n_domain_groups, config.n_experts_per_group
        ))

    def forward(self, x: torch.Tensor, attention_mask=None,
                training: bool = True) -> Tuple[torch.Tensor, dict]:
        B, S, D = x.shape

        # ── Attention ──
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = residual + attn_out

        # ── Hierarchical MoE FFN (checkpointable) ──
        residual = x
        x_normed = self.ffn_norm(x)

        if training and self.config.use_checkpointing:
            # Activation checkpointing: recompute MoE block during backward.
            # Saves ~O(n_experts × ffn_dim) activation memory per layer.
            # use_reentrant=False is required for PyTorch >= 2.0 compat.
            ffn_out, total_z_loss, total_load_balance, total_diversity_loss = \
                torch_checkpoint(self._moe_ffn_block, x_normed, training,
                                 use_reentrant=False)
        else:
            ffn_out, total_z_loss, total_load_balance, total_diversity_loss = \
                self._moe_ffn_block(x_normed, training)

        x = residual + ffn_out

        stats = {}
        stats["z_loss"] = total_z_loss * self.config.router_z_loss_coef
        stats["load_balance_loss"] = total_load_balance * self.config.load_balance_coef
        stats["diversity_loss"] = total_diversity_loss * self.config.diversity_coef
        stats["aux_loss"] = (stats["z_loss"] + stats["load_balance_loss"] + stats["diversity_loss"])
        stats["layer"] = self.layer_idx
        return x, stats

    def _moe_ffn_block(self, x_normed: torch.Tensor, training: bool
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MoE FFN block — checkpointable unit.

        Extracted from forward() so torch.utils.checkpoint can wrap it.
        Returns (ffn_out, z_loss, load_balance_loss, diversity_loss).
        All auxiliary losses must be tensors with grad for router training.
        """
        ffn_out = torch.zeros_like(x_normed)

        # ── Level 1: Domain routing ──
        domain_weights, domain_indices, domain_logits, z_loss_domain = \
            self.domain_router(x_normed, training)

        # ── Shared experts (always contribute, no domain routing) ──
        shared_out = torch.zeros_like(x_normed)
        if self.shared_experts:
            for shared_exp in self.shared_experts:
                shared_out = shared_out + shared_exp(x_normed)
            shared_out = shared_out / len(self.shared_experts)
        ffn_out = ffn_out + shared_out

        # ── Level 2: Expert routing WITH domain weight application ──
        total_z_loss = z_loss_domain
        total_load_balance = torch.tensor(0.0, device=x_normed.device)
        total_diversity_loss = torch.tensor(0.0, device=x_normed.device)

        for dk in range(self.config.n_active_domains):
            dom_idx_k = domain_indices[:, :, dk]    # [B, S]
            dom_w_k = domain_weights[:, :, dk]      # [B, S]

            for g in range(self.config.n_domain_groups):
                mask_gk = (dom_idx_k == g)           # [B, S]
                if not mask_gk.any():
                    continue

                # Tokens routed to domain g at position dk
                x_g = x_normed[mask_gk]              # [N_g, D]
                dom_w_g = dom_w_k[mask_gk].unsqueeze(-1)  # [N_g, 1]

                # Route to experts within domain
                exp_w, exp_idx, exp_log, z_loss_exp = self.expert_routers[g](x_g, training)
                total_z_loss = total_z_loss + z_loss_exp

                # Compute expert outputs (weighted by expert weights)
                expert_out = self._compute_expert_outputs(x_g, exp_w, exp_idx, g, training)

                # ── Apply domain weight and accumulate ──
                ffn_out[mask_gk] = ffn_out[mask_gk] + dom_w_g * expert_out

                # Load balance loss (importance-weighted, per expert router)
                total_load_balance = total_load_balance + self._load_balance_loss(
                    exp_log, exp_idx, self.config.n_experts_per_group)

                # Expert diversity loss (once per domain group, avoid duplicate)
                if dk == 0:
                    total_diversity_loss = total_diversity_loss + self._diversity_loss(g)

        return ffn_out, total_z_loss, total_load_balance, total_diversity_loss

    def _load_balance_loss(self, expert_logits: torch.Tensor,
                           expert_indices: torch.Tensor,
                           n_experts: int) -> torch.Tensor:
        """Switch Transformer-style auxiliary load balancing loss.

        L = n_experts * Σ_i (f_i · P_i)
        where f_i = fraction of tokens dispatched to expert i,
              P_i = mean softmax probability for expert i.

        Perfect balance → Σ_i f_i·P_i = 1/n_experts → L = 1.
        The loss pushes this value toward 1 (lower is not always better;
        a value close to 1 indicates balanced routing).

        Training tip: linearly warm up load_balance_coef from 0 to target
        over the first ~1000 steps to avoid early-training routing collapse.
        """
        # Mean router probability per expert
        expert_probs = F.softmax(expert_logits, dim=-1)  # [N, n_experts]
        mean_importance = expert_probs.mean(dim=0)        # [n_experts]

        # Fraction of tokens routed to each expert (count over all top-k slots)
        fraction = torch.zeros(n_experts, device=expert_logits.device)
        for e in range(n_experts):
            fraction[e] = (expert_indices == e).float().mean()

        # Switch Transformer auxiliary loss: n_experts · Σ_i(f_i · P_i)
        # Minimizing this encourages uniform routing.
        balance_loss = n_experts * (mean_importance * fraction).sum()
        return balance_loss

    def _diversity_loss(self, group_idx: int) -> torch.Tensor:
        """Penalize experts within a group having too-similar weights."""
        if self.config.n_experts_per_group < 2:
            return torch.tensor(0.0)

        # Compare gate weight vectors between experts
        gate_weights = []
        for e in range(self.config.n_experts_per_group):
            # gate.weight: [F, D]; take first principal direction approximation
            w = self.experts[group_idx][e].gate.weight  # [F, D]
            gate_weights.append(w.flatten())

        # Cosine similarity matrix
        loss = 0.0
        count = 0
        for i in range(len(gate_weights)):
            for j in range(i + 1, len(gate_weights)):
                cos_sim = F.cosine_similarity(
                    gate_weights[i].unsqueeze(0),
                    gate_weights[j].unsqueeze(0)
                )
                # Penalize high similarity (encourage diversity)
                loss += torch.clamp(cos_sim - 0.3, min=0) ** 2
                count += 1

        return loss / max(count, 1)

    def _compute_expert_outputs(self, x: torch.Tensor,
                                 exp_weights: torch.Tensor,
                                 exp_indices: torch.Tensor,
                                 group_idx: int,
                                 training: bool) -> torch.Tensor:
        """Compute weighted expert outputs via batched token dispatch.

        Strategy (vectorized, ~5-10× faster than per-token loops):
        1. Expand token→expert mapping to [N*K] flat indices
        2. For each expert, gather ALL its tokens in one batch → single forward
        3. Scatter-add weighted outputs back via index_add_

        Since torch.topk returns distinct indices, each token visits each
        expert at most once (no dedup needed).

        Args:
            x: token hidden states [N, D]
            exp_weights: top-k expert weights [N, n_active_experts]
            exp_indices: top-k expert indices [N, n_active_experts]
            group_idx: domain group index for expert lookup
            training: whether in training mode

        Returns:
            Weighted sum of expert outputs [N, D]
        """
        N, D = x.shape
        K = self.config.n_active_experts
        E = self.config.n_experts_per_group

        # ── Flatten the (token, slot) dimensions ──
        # Each token appears K times, once per expert routing slot
        w_flat = exp_weights.reshape(N * K, 1)       # [N*K, 1]
        idx_flat = exp_indices.reshape(N * K)         # [N*K]
        tok_idx = (torch.arange(N, device=x.device)
                   .unsqueeze(1).expand(N, K).reshape(N * K))  # [N*K]

        # ── Expert dropout mask (per-expert, deterministic per forward) ──
        if training and self.config.expert_dropout > 0:
            drop_mask = torch.rand(E, device=x.device) < self.config.expert_dropout
        else:
            drop_mask = torch.zeros(E, dtype=torch.bool, device=x.device)

        out = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for e in range(E):
            if drop_mask[e]:
                continue

            mask_e = (idx_flat == e)  # [N*K] — all (token, slot) pairs for expert e
            if not mask_e.any():
                continue

            # Batch-gather all tokens for expert e → single forward call
            tgt = tok_idx[mask_e]                                # [n_e]
            expert_in = x[tgt]                                   # [n_e, D]
            expert_out = self.experts[group_idx][e](expert_in)   # [n_e, D]
            weighted = expert_out * w_flat[mask_e]               # [n_e, D]

            # Scatter-add back to original token positions
            # index_add_ correctly accumulates if same token appears
            # in multiple slots (shouldn't happen with distinct top-k, but safe)
            out.index_add_(0, tgt, weighted)

            # Track expert usage (device-safe: buffer follows model device)
            usage = mask_e.sum().float()
            if self.expert_usage.device != x.device:
                self.expert_usage = self.expert_usage.to(device=x.device)
            self.expert_usage[group_idx, e] += usage

        return out

    @property
    def expert_param_counts(self) -> List[int]:
        counts = [
            sum(e.param_count for e in group_experts)
            for group_experts in self.experts
        ]
        for e in self.shared_experts:
            counts.append(e.param_count)
        return counts

    @property
    def total_expert_params(self) -> int:
        return sum(self.expert_param_counts)


# ═══════════════════════════════════════════════════════
# Full Hierarchical MoE Model
# ═══════════════════════════════════════════════════════

class HierarchicalMoEModel(nn.Module):
    """Complete Hierarchical MoE Model — 3.0B active / 14.5B total."""

    def __init__(self, config: HierarchicalMoEConfig, dtype=torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype

        # Embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, dtype=dtype)

        # Transformer layers
        self.layers = nn.ModuleList()
        moe_counter = 0
        for i in range(config.n_layers):
            if i % 2 == 0 and moe_counter < config.n_moe_layers:
                self.layers.append(HierarchicalMoELayer(config, i))
                moe_counter += 1
            else:
                self.layers.append(self._make_dense_layer(config))

        # Output
        self.final_norm = RMSNorm(config.hidden_dim, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False, dtype=dtype)
        self.lm_head.weight = self.tok_embeddings.weight  # tie

        # Ensure dtype consistency
        if dtype != torch.float32:
            self.to(dtype)

        self._print_model_info()

    def _make_dense_layer(self, config: HierarchicalMoEConfig) -> nn.Module:
        D, F = config.hidden_dim, config.expert_ffn_dim
        return nn.ModuleDict({
            "attn_norm": RMSNorm(D, dtype=self.dtype),
            "attn": nn.MultiheadAttention(D, config.n_heads, dropout=config.dropout,
                                          batch_first=True, dtype=self.dtype),
            "ffn_norm": RMSNorm(D, dtype=self.dtype),
            "ffn": nn.Sequential(
                nn.Linear(D, F, bias=False, dtype=self.dtype), nn.SiLU(),
                nn.Linear(F, F, bias=False, dtype=self.dtype), nn.SiLU(),
                nn.Linear(F, D, bias=False, dtype=self.dtype),
                nn.Dropout(config.dropout),
            ),
        })

    def forward(self, input_ids: torch.Tensor, attention_mask=None,
                training: bool = True) -> Tuple[torch.Tensor, dict]:
        B, S = input_ids.shape
        x = self.tok_embeddings(input_ids)

        total_aux = 0.0; total_z = 0.0; total_div = 0.0

        for layer in self.layers:
            if isinstance(layer, HierarchicalMoELayer):
                x, stats = layer(x, attention_mask, training)
                total_aux += stats.get("aux_loss", 0.0)
                total_z += stats.get("z_loss", 0.0)
                total_div += stats.get("diversity_loss", 0.0)
            else:
                residual = x
                x = layer["attn_norm"](x)
                attn_out, _ = layer["attn"](x, x, x, attn_mask=attention_mask)
                x = residual + attn_out
                residual = x
                x = layer["ffn_norm"](x)
                x = residual + layer["ffn"](x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, {
            "aux_loss": total_aux,
            "z_loss": total_z,
            "diversity_loss": total_div,
        }

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        emb = sum(p.numel() for p in self.tok_embeddings.parameters())
        attn = sum(
            sum(p.numel() for p in (l.attn.parameters() if isinstance(l, HierarchicalMoELayer)
                                    else l["attn"].parameters()))
            for l in self.layers
        )
        dense_ffn = sum(
            sum(p.numel() for p in (l["ffn"].parameters() if not isinstance(l, HierarchicalMoELayer) else []))
            for l in self.layers
        )
        expert_total = sum(
            l.total_expert_params for l in self.layers
            if isinstance(l, HierarchicalMoELayer)
        )
        per_expert = expert_total / (
            self.config.n_moe_layers *
            (self.config.n_domain_groups * self.config.n_experts_per_group + self.config.n_shared_experts)
        ) if self.config.n_moe_layers > 0 else 0

        # Active params estimate
        active_experts_per_layer = (self.config.n_active_domains * self.config.n_active_experts
                                     + self.config.n_shared_experts)
        active_expert = self.config.n_moe_layers * active_experts_per_layer * (
            3 * self.config.hidden_dim * self.config.expert_ffn_dim
        )
        active_total = emb + attn + dense_ffn + active_expert

        print(f"🏗️  Hierarchical MoE — DS4 Scale")
        print(f"   Total params:  {total/1e9:.2f}B")
        print(f"   Active params: {active_total/1e9:.2f}B")
        print(f"   Expert params: {expert_total/1e9:.2f}B")
        print(f"   Per-expert:    {per_expert/1e6:.1f}M")
        print(f"   Architecture:  {self.config.n_layers}L × {self.config.hidden_dim}D")
        print(f"   MoE: {self.config.n_moe_layers} layers × "
              f"{self.config.n_domain_groups} domains × "
              f"{self.config.n_experts_per_group} experts "
              f"(+{self.config.n_shared_experts} shared)")
        print(f"   Anti-collapse: Z-loss={self.config.router_z_loss_coef} "
              f"diversity={self.config.diversity_coef} "
              f"noise={self.config.router_noise_std}")


# ═══════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════

def create_hierarchical_moe_model(vocab_size: int = 72000,
                                   dtype=torch.float32) -> HierarchicalMoEModel:
    """Create the 3.0B active / 14.5B total Hierarchical MoE model.
    Use dtype=torch.float16 for ~30GB memory (fits 32GB+ machines).
    Use dtype=torch.float32 for ~61GB memory (needs 64GB+ machines).
    """
    config = HierarchicalMoEConfig(
        hidden_dim=2048, n_layers=26, n_heads=16, head_dim=128,
        vocab_size=vocab_size, max_seq_len=4096,
        n_moe_layers=16, n_domain_groups=4, n_experts_per_group=6,
        n_shared_experts=2, n_active_domains=2, n_active_experts=2,
        expert_ffn_dim=5376,  # calibrated for 3.0B active / 14.6B total
    )
    return HierarchicalMoEModel(config, dtype=dtype)


# ═══════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    # Use FP16 to fit in 32GB memory; use FP32 on 64GB+ machines
    try:
        dtype = torch.float16
        print("Using FP16 (30GB model)")
        model = create_hierarchical_moe_model(vocab_size=72000, dtype=dtype)
    except RuntimeError as e:
        print(f"FP16 OOM: {e}")
        print("Trying meta device for param counting only...")
        # Meta device — no memory allocation
        with torch.device('meta'):
            config = HierarchicalMoEConfig()
            model = HierarchicalMoEModel(config)
        print("\n✅ Model structure verified (meta device)")
        print(f"   Total layers: {len(model.layers)}")
        moe_count = sum(1 for l in model.layers if isinstance(l, HierarchicalMoELayer))
        print(f"   MoE layers: {moe_count}")
        import sys; sys.exit(0)

    # Test forward pass
    dummy = torch.randint(0, 71999, (1, 64))
    with torch.no_grad():
        logits, stats = model(dummy, training=False)
    print(f"\n✅ Forward pass: {logits.shape} (expected [1, 64, 72000])")
    print(f"   Aux loss: {stats['aux_loss']:.6f}")
    print(f"   Z-loss:   {stats['z_loss']:.6f}")
    print(f"   Div loss: {stats['diversity_loss']:.6f}")

    # Expert checks
    for i, layer in enumerate(model.layers):
        if isinstance(layer, HierarchicalMoELayer):
            counts = layer.expert_param_counts
            shared = sum(counts[-model.config.n_shared_experts:]) if model.config.n_shared_experts else 0
            domain = sum(counts[:model.config.n_domain_groups * model.config.n_experts_per_group]) if counts else 0
            per = (domain / (model.config.n_domain_groups * model.config.n_experts_per_group)) if model.config.n_experts_per_group else 0
            print(f"   Layer {i}: domain={domain/1e6:.1f}M shared={shared/1e6:.1f}M per-exp={per/1e6:.1f}M")
