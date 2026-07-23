"""
TinyLLM: Full Model Architecture.

Cutting-edge transformer with:
  - Multi-head Latent Attention (MLA)
  - Mixture of Experts (MoE) with load-balanced routing
  - SwiGLU activation
  - RMSNorm
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA) fallback
  - Multi-Token Prediction (MTP)
  - Sliding window attention
  - YaRN position interpolation
  - QK normalization (for large models)
  - Shared experts
  - Gradient checkpointing support
  - Full FSDP / DeepSpeed compatibility
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig, get_config
from .layers.attention import MultiHeadLatentAttention, GroupedQueryAttention
from .layers.moe import MoELayer, load_balancing_loss
from .layers.ffn import SwiGLUFFN
from .layers.normalization import RMSNorm
from .layers.mtp import MultiTokenPredictionHead


class TransformerLayer(nn.Module):
    """Single transformer block with optional MoE."""
    
    def __init__(
        self,
        layer_idx: int,
        config: ModelConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        D = config.hidden_dim
        
        # Pre-attention norm
        self.attn_norm = RMSNorm(D, config.norm_eps, dtype=dtype)
        
        # Attention
        if config.use_mla:
            self.attention = MultiHeadLatentAttention(
                hidden_dim=D,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                n_kv_heads=config.n_kv_heads,
                kv_latent_dim=config.kv_latent_dim,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                sliding_window=config.sliding_window,
                dropout=config.attn_dropout,
                use_qk_norm=config.use_qk_norm,
                dtype=dtype,
            )
        else:
            self.attention = GroupedQueryAttention(
                hidden_dim=D,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                n_kv_heads=config.n_kv_heads_effective,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                sliding_window=config.sliding_window,
                dropout=config.attn_dropout,
                dtype=dtype,
            )
        
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(D, config.norm_eps, dtype=dtype)
        
        # FFN: MoE or dense
        use_moe = config.use_moe and (
            not config.moe_layers or layer_idx in config.moe_layers
        )
        
        if use_moe:
            self.moe = MoELayer(
                hidden_dim=D,
                n_experts=config.n_experts,
                n_active=config.n_active_experts,
                expert_inter_dim=config.expert_inter_dim,
                capacity_factor=config.expert_capacity_factor,
                n_shared_experts=config.shared_experts,
                dropout=config.dropout,
                dtype=dtype,
            )
            self.use_moe = True
        else:
            self.ffn = SwiGLUFFN(D, config.ffn_inter_dim, dtype=dtype)
            self.use_moe = False
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
    
    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_gradient_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            hidden: [batch, seq_len, hidden_dim]
            aux_loss: MoE auxiliary loss (0 if dense FFN)
        """
        # Self-attention with residual
        residual = hidden
        hidden = self.attn_norm(hidden)
        
        if use_gradient_checkpointing and self.training:
            attn_out, _ = checkpoint(
                self.attention, hidden, position_ids, attention_mask, use_cache,
                use_reentrant=False,
            )
        else:
            attn_out, _ = self.attention(
                hidden, position_ids, attention_mask, use_cache=use_cache
            )
        
        hidden = residual + self.dropout(attn_out)
        
        # FFN with residual
        residual = hidden
        hidden = self.ffn_norm(hidden)
        
        aux_loss = torch.tensor(0.0, device=hidden.device)
        
        if self.use_moe:
            if use_gradient_checkpointing and self.training:
                ffn_out, aux_loss = checkpoint(
                    self.moe, hidden, use_reentrant=False,
                )
            else:
                ffn_out, aux_loss = self.moe(hidden)
        else:
            if use_gradient_checkpointing and self.training:
                ffn_out = checkpoint(self.ffn, hidden, use_reentrant=False)
            else:
                ffn_out = self.ffn(hidden)
        
        hidden = residual + self.dropout(ffn_out)
        
        return hidden, aux_loss


class TinyLLMModel(nn.Module):
    """
    TinyLLM: Cutting-edge MoE transformer model.
    
    Architecture:
        Embedding → [TransformerLayer × N] → FinalNorm → LM Head
                                ↓
                        MTP Heads (auxiliary)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        
        # Determine dtype
        if dtype is None:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map.get(config.train_dtype, torch.float32)
        self.dtype = dtype
        
        D = config.hidden_dim
        V = config.vocab_size
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(V, D, dtype=dtype)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(layer_idx=i, config=config, dtype=dtype)
            for i in range(config.n_layers)
        ])
        
        # Final norm
        self.final_norm = RMSNorm(D, config.norm_eps, dtype=dtype)
        
        # LM head (output projection)
        if config.tie_word_embeddings:
            self.lm_head_weight = self.tok_embeddings.weight
        else:
            self.lm_head = nn.Linear(D, V, bias=False, dtype=dtype)
        
        # Multi-Token Prediction heads
        if config.use_mtp and config.mtp_depth > 0:
            self.mtp_head = MultiTokenPredictionHead(
                hidden_dim=D,
                vocab_size=V,
                depth=config.mtp_depth,
                dtype=dtype,
            )
        else:
            self.mtp_head = None
        
        # Embedding multiplier (Gemma-style)
        self.embedding_multiplier = config.embedding_multiplier
        
        self._init_weights()
        self._print_model_info()
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        std = self.config.init_std
        
        for name, param in self.named_parameters():
            if "tok_embeddings" in name:
                nn.init.normal_(param, std=std)
            elif "lm_head" in name or "output" in name:
                nn.init.normal_(param, std=std / math.sqrt(2 * self.config.n_layers))
            elif "w_gate" in name or "w_up" in name or "w_down" in name:
                nn.init.normal_(param, std=std)
            elif "w_q" in name or "w_k" in name or "w_v" in name or "w_o" in name:
                nn.init.normal_(param, std=std)
            elif "weight" in name and param.dim() >= 2:
                nn.init.normal_(param, std=std)
            elif "weight" in name and param.dim() == 1:
                nn.init.ones_(param)
    
    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"╔══ TinyLLM: {self.config.name} ══╗")
        print(f"║ Hidden dim:   {self.config.hidden_dim:>8}        ║")
        print(f"║ Layers:       {self.config.n_layers:>8}        ║")
        print(f"║ Heads:        {self.config.n_heads:>8}        ║")
        print(f"║ Vocab:        {self.config.vocab_size:>8}        ║")
        if self.config.use_moe:
            print(f"║ MoE:          {self.config.n_experts}x{self.config.n_active_experts:>3}     ║")
        if self.config.use_mla:
            print(f"║ MLA latent:   {self.config.kv_latent_dim:>8}        ║")
        if self.config.use_mtp and self.mtp_head:
            print(f"║ MTP depth:    {self.config.mtp_depth:>8}        ║")
        print(f"║ Total params: {total/1e9:>7.2f}B      ║")
        print(f"║ Trainable:    {trainable/1e9:>7.2f}B      ║")
        print(f"║ Dtype:        {str(self.dtype):>13} ║")
        print("╚══════════════════════════════════╝")
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_embeddings
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.tok_embeddings = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        use_gradient_checkpointing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] or [batch, 1, seq_len, seq_len]
            position_ids: [batch, seq_len]
            labels: [batch, seq_len] for loss computation
            use_cache: return KV cache for inference
        
        Returns:
            dict with keys: logits, loss, aux_loss, hidden_states (optional)
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        # Causal attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, T] → [B, 1, T, T]
                causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = causal & pad_mask
                attn_mask = (1.0 - attn_mask.float()) * -1e9
            else:
                attn_mask = attention_mask
        else:
            attn_mask = None
        
        # Embedding
        hidden = self.tok_embeddings(input_ids)
        if self.embedding_multiplier != 1.0:
            hidden = hidden * self.embedding_multiplier
        
        all_hidden_states = [hidden] if output_hidden_states else None
        total_aux_loss = torch.tensor(0.0, device=device)
        
        # Transformer layers
        for layer in self.layers:
            hidden, aux_loss = layer(
                hidden,
                position_ids=position_ids,
                attention_mask=attn_mask,
                use_cache=use_cache,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            total_aux_loss = total_aux_loss + aux_loss
            
            if output_hidden_states:
                all_hidden_states.append(hidden)
        
        # Final norm
        hidden = self.final_norm(hidden)
        
        # LM head
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.lm_head_weight)
        else:
            logits = self.lm_head(hidden)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # MTP loss
            mtp_loss = torch.tensor(0.0, device=device)
            if self.mtp_head is not None and self.training:
                input_emb = self.tok_embeddings(input_ids)
                mtp_logits = self.mtp_head(hidden, input_emb)
                mtp_loss = self.mtp_head.compute_loss(
                    mtp_logits, input_ids[:, :-1], labels,
                )
            
            # Total loss
            loss = ce_loss + 0.3 * mtp_loss + self.config.moe_aux_loss_weight * total_aux_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "hidden_states": all_hidden_states,
            "last_hidden_state": hidden,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        B = input_ids.shape[0]
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]
            
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, -1:]] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop on EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all layers."""
        self._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def create_model(
    config_name: str = "small",
    dtype: Optional[torch.dtype] = None,
) -> TinyLLMModel:
    """Factory function to create a TinyLLM model from a named config."""
    config = get_config(config_name)
    return TinyLLMModel(config, dtype=dtype)


def list_models():
    """List all available model configurations."""
    from .config import list_configs
    return list_configs()
