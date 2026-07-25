"""
Mixture of Experts (MoE) with load balancing.

Supports:
  - Top-k expert routing with softmax gating
  - Shared experts (always active)
  - Expert capacity limiting (token dropping prevention)
  - Load balancing auxiliary loss
  - Expert parallelism ready design
  - Z-loss regularization for training stability
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLUFFN


class MoEGate(nn.Module):
    """Learned routing gate: selects top-k experts per token.
    
    Features for training stability:
      - Noisy gating: adds Gaussian noise during training to prevent expert collapse
      - Z-loss regularization: penalizes large logit magnitudes
      - Load balancing auxiliary loss
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_experts: int,
        n_active: int = 2,
        capacity_factor: float = 1.25,
        gate_noise: float = 0.1,         # ← noise sigma during training
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise
        
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_dim, dtype=dtype))
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.02 / math.sqrt(self.weight.shape[1]))
    
    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch * seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        
        Returns:
            topk_indices: [tokens, n_active] — expert indices per token
            topk_weights: [tokens, n_active] — normalized gate weights
            aux_loss: scalar — load balancing loss
        """
        # Flatten to [total_tokens, hidden_dim]
        if hidden.dim() == 3:
            B, T, D = hidden.shape
            hidden = hidden.view(B * T, D)
        
        # Compute gate logits
        logits = F.linear(hidden, self.weight)  # [tokens, n_experts]
        
        # 🔄 Noisy gating: add Gaussian noise during training
        # This prevents the router from collapsing to a single expert.
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(logits) * self.gate_noise
            logits = logits + noise
        
        # Z-loss for training stability (DeepSeek-V3)
        z_loss = logits.logsumexp(dim=-1).pow(2).mean() * 1e-3
        
        # Top-k selection
        topk_logits, topk_indices = torch.topk(logits, self.n_active, dim=-1)
        
        # Softmax over selected experts
        topk_weights = F.softmax(topk_logits, dim=-1)
        
        # Load balancing loss
        aux_loss = load_balancing_loss(logits, topk_indices, self.n_experts)
        
        return topk_indices, topk_weights, aux_loss + z_loss


def load_balancing_loss(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss.
    Encourages uniform expert utilization.
    
    L_aux = n_experts * sum(f_i * P_i)
    where f_i = fraction of tokens dispatched to expert i
          P_i = average softmax probability for expert i
    """
    # One-hot of top-k indices
    mask = F.one_hot(topk_indices, n_experts).float()  # [tokens, k, n_experts]
    
    # Fraction of tokens per expert: f_i
    density = mask.mean(dim=0).sum(dim=0)  # [n_experts]
    
    # Average router probability per expert: P_i
    probs = F.softmax(logits, dim=-1)  # [tokens, n_experts]
    density_proxy = probs.mean(dim=0)  # [n_experts]
    
    loss = (density * density_proxy).sum() * n_experts
    return loss


class ExpertFFN(nn.Module):
    """Single expert FFN with SwiGLU activation."""
    
    def __init__(
        self,
        hidden_dim: int,
        inter_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ffn = SwiGLUFFN(hidden_dim, inter_dim, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoELayer(nn.Module):
    """
    Sparsely-gated Mixture of Experts layer.
    
    Tokens are routed to top-k experts via a learned gate.
    Only active experts' FFNs are computed → massive compute savings.
    
    Architecture:
      gate(hidden) → top-k expert indices + weights
      for each expert e:
          mask = [tokens routed to e]
          expert_out = ExpertFFN_e(hidden[mask])
      output = sum(weight_e * expert_out_e) across top-k
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_experts: int,
        n_active: int = 2,
        expert_inter_dim: int = 1024,
        capacity_factor: float = 1.25,
        n_shared_experts: int = 0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.n_active = n_active
        
        # Router gate
        self.gate = MoEGate(
            hidden_dim, n_experts, n_active,
            capacity_factor=capacity_factor,
            dtype=dtype
        )
        
        # Expert FFNs
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim, expert_inter_dim, dtype)
            for _ in range(n_experts)
        ])
        
        # Shared experts (always active, applied to all tokens)
        self.n_shared = n_shared_experts
        if n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertFFN(hidden_dim, expert_inter_dim, dtype)
                for _ in range(n_shared_experts)
            ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: MoE auxiliary loss
        """
        B, T, D = hidden.shape
        tokens = hidden.view(B * T, D)
        n_tokens = tokens.shape[0]
        
        # 1. Route tokens to experts
        topk_indices, topk_weights, aux_loss = self.gate(tokens)
        # topk_indices: [tokens, k], topk_weights: [tokens, k]
        
        # 2. Capacity enforcement
        capacity = max(1, int(math.ceil(self.gate.capacity_factor * n_tokens / self.n_experts)))
        
        # 3. Optimized dispatch: group tokens by expert, process in batches
        #    Instead of O(k * n_experts) double loop, we:
        #    a) Flatten (token, k) → single dispatch list
        #    b) Group by expert_id
        #    c) Process each expert's batch in one forward call
        #    d) Scatter-add weighted outputs back
        
        output = torch.zeros_like(tokens)
        
        # Build flat dispatch: for each (token, k_rank), record expert + weight
        token_idx = torch.arange(n_tokens, device=tokens.device).unsqueeze(1).expand(-1, self.n_active)
        # token_idx: [tokens, k] — original token position
        
        # Sort by expert_id for grouped processing
        flat_experts = topk_indices.reshape(-1)        # [tokens * k]
        flat_weights = topk_weights.reshape(-1)         # [tokens * k]
        flat_tokens  = token_idx.reshape(-1)            # [tokens * k]
        
        # For each expert, gather all tokens routed to it
        for eid in range(self.n_experts):
            expert_mask = (flat_experts == eid)
            n_routed = expert_mask.sum().item()
            if n_routed == 0:
                continue
            
            # Capacity enforcement
            if n_routed > capacity:
                # Keep first `capacity` tokens, drop rest
                keep = torch.where(expert_mask)[0][:capacity]
                expert_mask = torch.zeros_like(expert_mask)
                expert_mask[keep] = True
                n_routed = capacity
            
            # Gather tokens for this expert
            tok_indices = flat_tokens[expert_mask]       # [n_routed]
            tok_weights = flat_weights[expert_mask]      # [n_routed]
            expert_input = tokens[tok_indices]            # [n_routed, D]
            
            # Single batched forward for all tokens routed to this expert
            expert_output = self.experts[eid](expert_input)  # [n_routed, D]
            
            # Weight and scatter-add back
            weighted = expert_output * tok_weights.unsqueeze(-1)
            output.index_add_(0, tok_indices, weighted)
        
        # 4. Shared experts (always active)
        if self.n_shared > 0:
            shared_out = sum(exp(tokens) for exp in self.shared_experts) / self.n_shared
            output = output + shared_out
        
        output = self.dropout(output)
        output = output.view(B, T, D)
        
        return output, aux_loss


def compute_expert_metrics(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    n_experts: int,
) -> dict:
    """Compute monitoring metrics for MoE router health.
    
    Call this during training to log:
      - expert_token_count: tokens per expert
      - avg_entropy: average gate entropy (higher = more diverse routing)
      - imbalance_ratio: max(count) / mean(count), 1.0 = perfectly balanced
      - pct_top5: % tokens assigned to top 5 experts
    """
    with torch.no_grad():
        # Per-expert token counts
        counts = torch.zeros(n_experts, dtype=torch.float32, device=topk_indices.device)
        for k in range(topk_indices.shape[1]):
            idx = topk_indices[:, k]
            counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        
        # Gate entropy (higher = more diverse routing)
        probs = F.softmax(topk_weights, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        
        # Imbalance ratio (1.0 = perfect, high = collapse)
        mean_count = counts.mean().clamp(min=1)
        imbalance = counts.max() / mean_count
        
        # % tokens going to top-5 experts
        sorted_counts, _ = counts.sort(descending=True)
        pct_top5 = sorted_counts[:5].sum() / counts.sum().clamp(min=1) * 100
        
        return {
            'expert_token_counts': counts.tolist(),
            'avg_gate_entropy': entropy.item(),
            'imbalance_ratio': imbalance.item(),
            'pct_top5_experts': pct_top5.item(),
        }
