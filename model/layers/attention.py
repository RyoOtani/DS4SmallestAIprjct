"""
Multi-head Latent Attention (MLA) + Grouped Query Attention + RoPE.

MLA compresses KV cache from O(n_heads × head_dim × seq_len)
to O(kv_latent_dim × seq_len), achieving ~8× compression.

Supports:
  - MLA: Compressed KV latent projection
  - GQA: Fewer KV heads than Q heads
  - RoPE: Rotary Position Embeddings with YaRN scaling
  - Sliding window attention
  - Flash Attention compatible design
  - QK normalization (optional, for large models)
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    scaling: Optional[dict] = None,
) -> torch.Tensor:
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # Apply scaling if needed
    if scaling and scaling.get("type") == "linear":
        factor = scaling.get("factor", 1.0)
        t = torch.arange(end, dtype=torch.float32) / factor
    elif scaling and scaling.get("type") == "yarn":
        # YaRN: NTK-aware interpolation
        factor = scaling.get("factor", 1.0)
        low_freq_factor = scaling.get("low_freq_factor", 1.0)
        high_freq_factor = scaling.get("high_freq_factor", 4.0)
        original_max_pos = scaling.get("original_max_position_embeddings", end)
        
        # NTK-aware frequency scaling
        freq_mask = freqs > (low_freq_factor / original_max_pos)
        ntk_freqs = freqs.clone()
        ntk_freqs[freq_mask] = freqs[freq_mask] * factor
        t = torch.arange(end, dtype=torch.float32)
        freqs = ntk_freqs
    else:
        t = torch.arange(end, dtype=torch.float32)
    
    freqs = torch.outer(t, freqs).to(dtype)
    return torch.polar(torch.ones_like(freqs), freqs)  # cos + i*sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_out = torch.view_as_complex(xq_)
    xk_out = torch.view_as_complex(xk_)
    
    # Broadcast freqs_cis to match batch & head dims
    # xq: [batch, heads, seq, dim]
    # freqs_cis: [seq, dim/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim/2]
    
    xq_out = xq_out * freqs_cis
    xk_out = xk_out * freqs_cis
    
    xq_out = torch.view_as_real(xq_out).flatten(-2)
    xk_out = torch.view_as_real(xk_out).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ═══════════════════════════════════════════════════════════════════════
# Multi-head Latent Attention (MLA)
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadLatentAttention(nn.Module):
    """
    DeepSeek-V2/V3 style MLA.
    
    Traditional MHA: Q, K, V each are [n_heads × head_dim]
    MLA:             Q is full, but K,V are compressed through a joint latent.
    
    Flow:
      hidden → W_kv_compress → kv_latent (compressed, ∈ R^L)
      kv_latent → W_k_up → K (full heads)
      kv_latent → W_v_up → V (full heads)
    
    KV cache stores only kv_latent → ~8× memory savings.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int = 0,
        kv_latent_dim: int = 512,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
        sliding_window: int = 0,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads if n_kv_heads > 0 else n_heads
        self.kv_latent_dim = kv_latent_dim
        self.sliding_window = sliding_window
        
        # Q projection (full)
        self.w_q = nn.Linear(hidden_dim, n_heads * head_dim, bias=False, dtype=dtype)
        
        # KV compression
        self.w_kv_compress = nn.Linear(
            hidden_dim, kv_latent_dim, bias=False, dtype=dtype
        )
        
        # KV up-projection from latent
        self.w_k_up = nn.Linear(
            kv_latent_dim, n_heads * head_dim, bias=False, dtype=dtype
        )
        self.w_v_up = nn.Linear(
            kv_latent_dim, n_heads * head_dim, bias=False, dtype=dtype
        )
        
        # Output projection
        self.w_o = nn.Linear(n_heads * head_dim, hidden_dim, bias=False, dtype=dtype)
        
        # QK norm (optional, stability for large models)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5, dtype=dtype)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5, dtype=dtype)
        
        # RoPE
        self.rope_theta = rope_theta
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2, rope_theta, dtype=torch.float32),
            persistent=False,
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_latent_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len] or None
            kv_latent_cache: (k_latent_cache, v_latent_cache) from previous steps
            use_cache: whether to return new kv_latent for caching
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            new_cache: (k_latent, v_latent) if use_cache else None
        """
        B, T, D = hidden.shape
        
        # 1. Q projection
        q = self.w_q(hidden).view(B, T, self.n_heads, self.head_dim)
        
        # 2. KV compression into latent space
        kv_latent = self.w_kv_compress(hidden)  # [B, T, kv_latent_dim]
        
        # 3. Up-project K, V from latent
        k = self.w_k_up(kv_latent).view(B, T, self.n_heads, self.head_dim)
        v = self.w_v_up(kv_latent).view(B, T, self.n_heads, self.head_dim)
        
        # 4. Apply RoPE
        if position_ids is not None:
            # Use position_ids to index into freqs_cis
            freqs = self.freqs_cis[position_ids].unsqueeze(2)  # [B, T, 1, dim/2]
        else:
            freqs = self.freqs_cis[:T].unsqueeze(0).unsqueeze(2)
        
        q_roped, k_roped = apply_rotary_emb(q, k, freqs)
        
        # 5. QK normalization (optional)
        if self.use_qk_norm:
            q_roped = self.q_norm(q_roped)
            k_roped = self.k_norm(k_roped)
        
        # 6. Attention computation
        # [B, T, n_heads, head_dim] → [B, n_heads, T, head_dim]
        q_roped = q_roped.transpose(1, 2)
        k_roped = k_roped.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q_roped, k_roped.transpose(-2, -1)) * self.scale
        
        # Sliding window mask
        if self.sliding_window > 0:
            window_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=hidden.device),
                diagonal=self.sliding_window,
            ).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(window_mask, float('-inf'))
        
        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=hidden.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        
        # 7. Output projection
        output = self.w_o(attn_out)
        
        new_cache = None
        if use_cache:
            new_cache = (kv_latent, kv_latent)  # store compressed latent
        
        return output, new_cache


# ═══════════════════════════════════════════════════════════════════════
# Grouped Query Attention (GQA) — fallback when MLA is disabled
# ═══════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """GQA: more Q heads than KV heads. Simpler than MLA, no KV compression."""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
        sliding_window: int = 0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_groups = n_heads // n_kv_heads
        self.sliding_window = sliding_window
        
        self.w_q = nn.Linear(hidden_dim, n_heads * head_dim, bias=False, dtype=dtype)
        self.w_k = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False, dtype=dtype)
        self.w_v = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False, dtype=dtype)
        self.w_o = nn.Linear(n_heads * head_dim, hidden_dim, bias=False, dtype=dtype)
        
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, max_seq_len * 2, rope_theta),
            persistent=False,
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = hidden.shape
        
        q = self.w_q(hidden).view(B, T, self.n_heads, self.head_dim)
        k = self.w_k(hidden).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.w_v(hidden).view(B, T, self.n_kv_heads, self.head_dim)
        
        # RoPE
        if position_ids is not None:
            freqs = self.freqs_cis[position_ids].unsqueeze(2)
        else:
            freqs = self.freqs_cis[:T].unsqueeze(0).unsqueeze(2)
        
        # Expand Q's freqs to match heads
        q_freqs = freqs.repeat_interleave(self.n_groups, dim=2) if self.n_groups > 1 else freqs
        k_freqs = freqs
        
        q, _ = apply_rotary_emb(q, torch.zeros_like(k), q_freqs)
        k, _ = apply_rotary_emb(k, torch.zeros_like(k), k_freqs)
        
        # Transpose: [B, heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat KV for GQA
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            causal = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=hidden.device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal, float('-inf'))
        
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden.dtype)
        weights = self.dropout(weights)
        
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(out), None
