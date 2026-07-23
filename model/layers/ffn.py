"""SwiGLU FFN and Gated MLP layers."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    FFN(x) = W_down @ (SiLU(W_gate @ x) * W_up @ x)
    
    This is the standard FFN used in LLaMA, DeepSeek, Mixtral, etc.
    More expressive than ReLU-FFN at the same parameter count.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        inter_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Ensure inter_dim is a multiple of 256 (efficiency)
        inter_dim = ((inter_dim + 255) // 256) * 256
        
        self.w_gate = nn.Linear(hidden_dim, inter_dim, bias=False, dtype=dtype)
        self.w_up = nn.Linear(hidden_dim, inter_dim, bias=False, dtype=dtype)
        self.w_down = nn.Linear(inter_dim, hidden_dim, bias=False, dtype=dtype)
        
        self._init_weights()
    
    def _init_weights(self):
        for w in [self.w_gate, self.w_up, self.w_down]:
            nn.init.normal_(w.weight, std=0.02 / (2 * w.weight.shape[0] / w.weight.shape[1]) ** 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = F.silu(self.w_gate(x))
        up_out = self.w_up(x)
        return self.w_down(gate_out * up_out)


class GatedMLP(nn.Module):
    """Alternative gated MLP with configurable activation."""
    
    ACTIVATIONS = {
        "swiglu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
        "swish": F.silu,
    }
    
    def __init__(
        self,
        hidden_dim: int,
        inter_dim: int,
        activation: str = "swiglu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        inter_dim = ((inter_dim + 255) // 256) * 256
        self.activation_fn = self.ACTIVATIONS.get(activation, F.silu)
        
        self.w_gate = nn.Linear(hidden_dim, inter_dim, bias=False, dtype=dtype)
        self.w_up = nn.Linear(hidden_dim, inter_dim, bias=False, dtype=dtype)
        self.w_down = nn.Linear(inter_dim, hidden_dim, bias=False, dtype=dtype)
        
        self._init_weights()
    
    def _init_weights(self):
        for w in [self.w_gate, self.w_up, self.w_down]:
            nn.init.normal_(w.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.activation_fn(self.w_gate(x)) * self.w_up(x))
