"""
Multi-Token Prediction (MTP) Heads.

Instead of predicting only the next token, MTP predicts the next N tokens
simultaneously using independent output heads. This improves sample efficiency
and knowledge density — a key DeepSeek-V3 innovation.

Each MTP head h_k predicts the token at position t + k given the hidden state
at position t. Heads share the embedding layer but have independent projections.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm
from .ffn import SwiGLUFFN


class MTPHead(nn.Module):
    """Single MTP prediction head for one future position."""
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int = 1,  # transformer sub-layers in this head
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Shared input projection
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False, dtype=dtype)
        
        # Optional transformer sub-layers
        self.sub_layers = nn.ModuleList()
        for _ in range(depth):
            self.sub_layers.append(nn.ModuleDict({
                "norm": RMSNorm(hidden_dim, dtype=dtype),
                "ffn": SwiGLUFFN(hidden_dim, hidden_dim * 4, dtype=dtype),
            }))
        
        # Final norm + output
        self.final_norm = RMSNorm(hidden_dim, dtype=dtype)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=dtype)
    
    def forward(
        self,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim] — current layer's hidden state
            embedding: [batch, seq_len, hidden_dim] — token embedding of previous position
        
        Returns:
            logits: [batch, seq_len, vocab_size] — predictions for t+k
        """
        # Combine hidden state with token embedding
        combined = torch.cat([hidden, embedding], dim=-1)
        x = self.input_proj(combined)
        
        for sub in self.sub_layers:
            residual = x
            x = sub["norm"](x)
            x = sub["ffn"](x)
            x = x + residual
        
        x = self.final_norm(x)
        return self.output(x)


class MultiTokenPredictionHead(nn.Module):
    """
    Full MTP module with K prediction heads.
    
    Each head k predicts the token at position t + k + 1.
    Head 0 = standard next-token prediction (equivalent to lm_head).
    Heads 1..K = additional future predictions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.depth = depth
        
        self.heads = nn.ModuleList([
            MTPHead(hidden_dim, vocab_size, depth=1, dtype=dtype)
            for _ in range(depth)
        ])
        
        # Embedding norm for combining with hidden
        self.emb_norm = RMSNorm(hidden_dim, dtype=dtype)
    
    def forward(
        self,
        hidden: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
            input_embeddings: [batch, seq_len, hidden_dim]
        
        Returns:
            List of logits, one per prediction head: [head_0, head_1, ..., head_{K-1}]
        """
        emb_normed = self.emb_norm(input_embeddings)
        outputs = []
        
        current_hidden = hidden
        for head in self.heads:
            logits = head(current_hidden, emb_normed)
            outputs.append(logits)
            # Chain: each head's output representation feeds into next
            current_hidden = head.input_proj(
                torch.cat([current_hidden, emb_normed], dim=-1)
            )
        
        return outputs
    
    def compute_loss(
        self,
        logits_list: list[torch.Tensor],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Compute MTP loss over all heads.
        
        Args:
            logits_list: K heads × [batch, seq_len, vocab_size]
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] — shifted by 1
        """
        total_loss = torch.tensor(0.0, device=input_ids.device)
        
        for k, logits in enumerate(logits_list):
            # Target for head k: labels shifted by k
            if k == 0:
                target = labels[:, :logits.shape[1]]
            else:
                target = input_ids[:, k:logits.shape[1]+k]
            
            logits_flat = logits.reshape(-1, logits.shape[-1])
            target_flat = target.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, target_flat, ignore_index=ignore_index)
            total_loss = total_loss + loss
        
        return total_loss / (k + 1)
