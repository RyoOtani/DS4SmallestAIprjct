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
        self.gate = MoEGate(hidden_dim, n_experts, n_active, capacity_factor, dtype)
        
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
        
        # 2. Capacity enforcement (per expert)
        # Each expert can process at most `capacity` tokens.
        # Overflow tokens are rerouted to their next-best expert (top-2 → top-1 fallback).
        capacity = max(1, int(math.ceil(self.gate.capacity_factor * n_tokens / self.n_experts)))
        
        # 3. Dispatch and compute with capacity
        output = torch.zeros_like(tokens)
        overflow_mask = torch.ones(n_tokens, dtype=torch.bool, device=tokens.device)
        
        for k in range(self.n_active):
            expert_ids = topk_indices[:, k]  # [B*T]
            weights = topk_weights[:, k].unsqueeze(-1)  # [B*T, 1]
            
            for eid in range(self.n_experts):
                mask = (expert_ids == eid) & overflow_mask
                n_selected = mask.sum().item()
                if n_selected == 0:
                    continue
                
                # 🔄 Capacity enforcement: drop tokens beyond capacity
                if n_selected > capacity:
                    # Keep only the first `capacity` tokens (by order in batch)
                    selected_indices = torch.where(mask)[0]
                    keep_indices = selected_indices[:capacity]
                    drop_indices = selected_indices[capacity:]
                    
                    # Mark dropped tokens for reroute
                    mask[drop_indices] = False
                    # These tokens will be picked up by next k (next-best expert)
                
                expert_out = self.experts[eid](tokens[mask])
                output[mask] += expert_out * weights[mask]
                overflow_mask[mask] = False  # mark as processed
        
        # 4. Handle residual tokens (exceeded all capacity)
        # Route them uniformly across all shared experts or pass through
        if overflow_mask.any():
            # Simple fallback: average of all experts for overflow tokens
            residual = tokens[overflow_mask]
            if self.n_shared > 0:
                fallback = sum(exp(residual) for exp in self.shared_experts) / self.n_shared
            else:
                # Identity-ish pass-through (just a tiny transform)
                fallback = residual * 0.99
            output[overflow_mask] += fallback
        
        # 5. Shared experts (always active)
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
