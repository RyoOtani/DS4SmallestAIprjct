"""Normalization layers: RMSNorm, QKNorm, DeepNorm."""

from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    y = x / sqrt(mean(x^2) + eps) * weight
    
    Faster than LayerNorm (no mean subtraction, no bias).
    Used in LLaMA, DeepSeek, Mixtral, Gemma.
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability
        input_dtype = x.dtype
        x = x.float()
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        
        return (x * self.weight.float()).to(input_dtype)


class QKNorm(nn.Module):
    """
    Q/K normalization for large model stability.
    
    Normalizes query and key before attention dot product to prevent
    attention logit explosion in very deep/large models.
    """
    
    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class DeepNorm(nn.Module):
    """
    DeepNet/DeepNorm initialization for ultra-deep transformers.
    
    Scales residual connections to ensure stable training
    for models with 100+ layers.
    
    alpha = (2 * N_layers)^(1/4) where N_layers is total depth.
    Residual is scaled by alpha before adding.
    """
    
    def __init__(
        self,
        dim: int,
        n_layers: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.alpha = (2.0 * n_layers) ** 0.25
        self.norm = RMSNorm(dim, eps, dtype)
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual * self.alpha + self.norm(x)
