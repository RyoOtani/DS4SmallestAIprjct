#!/usr/bin/env python3
"""
create_demo_model.py — Generate a tiny demo model with random weights for testing.

Produces:
  - demo-tinyllm-nano/  — HuggingFace format (model.pt + config.json + tokenizer/)
  - demo-tinyllm-nano.gguf  — GGUF format for C runtime (FP16, ~600 MB)

This enables end-to-end testing of the entire pipeline without needing
GPU training or downloading large models.

Usage:
  python create_demo_model.py                    # Create 1.5B param demo model
  python create_demo_model.py --size micro        # Create 100M param micro model (fast)
  python create_demo_model.py --size nano --q4    # Create Q4 quantized GGUF
"""

import os, sys, json, struct, time, argparse
import numpy as np
import torch
import torch.nn as nn

# Add repo root to path for export_gguf import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_demo_model(hidden_dim=1024, num_layers=24, vocab_size=32000,
                      intermediate_size=2816, num_heads=16, output_dir="demo-tinyllm-nano"):
    """Create a TinyLLM model with random weights and save it."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🧪 Creating demo TinyLLM model...")
    print(f"   hidden={hidden_dim}, layers={num_layers}, vocab={vocab_size}")
    print(f"   intermediate={intermediate_size}, heads={num_heads}")
    
    # ── Build model ────────────────────────────────────────
    class TinyLLMLayer(nn.Module):
        def __init__(self):
            super().__init__()
            D = hidden_dim
            I = intermediate_size
            self.norm1 = nn.RMSNorm(D, eps=1e-5)
            self.norm2 = nn.RMSNorm(D, eps=1e-5)
            self.q_proj = nn.Linear(D, D, bias=False)
            self.k_proj = nn.Linear(D, D, bias=False)
            self.v_proj = nn.Linear(D, D, bias=False)
            self.o_proj = nn.Linear(D, D, bias=False)
            self.gate_proj = nn.Linear(D, I, bias=False)
            self.up_proj = nn.Linear(D, I, bias=False)
            self.down_proj = nn.Linear(I, D, bias=False)
        
        def forward(self, x):
            residual = x
            x = self.norm1(x)
            B, S, D = x.shape
            h, d = num_heads, D // num_heads
            q = self.q_proj(x).view(B, S, h, d).transpose(1, 2)
            k = self.k_proj(x).view(B, S, h, d).transpose(1, 2)
            v = self.v_proj(x).view(B, S, h, d).transpose(1, 2)
            a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = self.o_proj(a.transpose(1, 2).contiguous().view(B, S, D)) + residual
            residual = x
            x = self.norm2(x)
            x = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
            return x
    
    class TinyLLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([TinyLLMLayer() for _ in range(num_layers)])
            self.norm = nn.RMSNorm(hidden_dim, eps=1e-5)
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        def forward(self, input_ids, labels=None):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1), ignore_index=-100)
            return {'loss': loss, 'logits': logits}
    
    model = TinyLLMModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total/1e9:.2f}B")
    
    # Initialize with small random weights (Xavier-like)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_normal_(p, gain=0.5)
        else:
            nn.init.normal_(p, std=0.02)
    
    # ── Save HuggingFace format ────────────────────────────
    print(f"\n💾 Saving HuggingFace format to {output_dir}/")
    torch.save({'model_state_dict': model.state_dict(),
                'config': {'hidden_size': hidden_dim, 'num_hidden_layers': num_layers,
                          'vocab_size': vocab_size, 'intermediate_size': intermediate_size,
                          'num_attention_heads': num_heads, 'max_position_embeddings': 8192},
                'step': 0}, f'{output_dir}/model.pt')
    
    config = {
        'hidden_size': hidden_dim, 'num_hidden_layers': num_layers,
        'vocab_size': vocab_size, 'intermediate_size': intermediate_size,
        'num_attention_heads': num_heads, 'max_position_embeddings': 8192,
        'model_type': 'tinyllm-nano',
    }
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy tokenizer if available
    for src in ['tokenizer', '../tokenizer']:
        tok_json = os.path.join(src, 'tokenizer.json')
        if os.path.exists(tok_json):
            import shutil
            shutil.copy(tok_json, f'{output_dir}/tokenizer.json')
            cfg_src = os.path.join(src, 'tokenizer_config.json')
            if os.path.exists(cfg_src):
                shutil.copy(cfg_src, f'{output_dir}/tokenizer_config.json')
            print(f"   ✅ Tokenizer copied from {src}/")
            break
    else:
        print(f"   ⚠️  No tokenizer found — copy tokenizer/ manually")
    
    hf_mb = os.path.getsize(f'{output_dir}/model.pt') / (1024**2)
    print(f"   ✅ HF format: {hf_mb:.0f} MB")
    
    return model, config


def export_to_gguf_demo(model, config, output_path="demo-tinyllm-nano.gguf"):
    """Export model to GGUF format for C runtime testing."""
    from export_gguf import export_to_gguf, TinyLLMLayer as ELayer, TinyLLMModel as EModel
    
    print(f"\n🔧 Exporting GGUF: {output_path}")
    
    # Create export model matching the demo model's structure
    class ExportLayer(torch.nn.Module):
        def __init__(self, D, inter):
            super().__init__()
            self.norm1 = torch.nn.RMSNorm(D, eps=1e-5)
            self.norm2 = torch.nn.RMSNorm(D, eps=1e-5)
            self.q_proj = torch.nn.Linear(D, D, bias=False)
            self.k_proj = torch.nn.Linear(D, D, bias=False)
            self.v_proj = torch.nn.Linear(D, D, bias=False)
            self.o_proj = torch.nn.Linear(D, D, bias=False)
            self.gate_proj = torch.nn.Linear(D, inter, bias=False)
            self.up_proj = torch.nn.Linear(D, inter, bias=False)
            self.down_proj = torch.nn.Linear(inter, D, bias=False)
        def forward(self, x):
            raise NotImplementedError("Export only — use C runtime for inference")
    
    class ExportModel(torch.nn.Module):
        def __init__(self, V, D, L, I):
            super().__init__()
            self.embed = torch.nn.Embedding(V, D)
            self.layers = torch.nn.ModuleList([ExportLayer(D, I) for _ in range(L)])
            self.norm = torch.nn.RMSNorm(D, eps=1e-5)
            self.lm_head = torch.nn.Linear(D, V, bias=False)
        def forward(self, x):
            raise NotImplementedError("Export only — use C runtime for inference")
    
    D = config['hidden_size']
    V = config['vocab_size']
    L = config['num_hidden_layers']
    I = config['intermediate_size']
    
    export_model = ExportModel(V, D, L, I)
    
    # Copy weights from demo model
    demo_sd = model.state_dict()
    export_model.load_state_dict(demo_sd, strict=True)
    
    export_to_gguf(export_model, output_path, config, use_q4=False)
    
    gguf_mb = os.path.getsize(output_path) / (1024**2)
    print(f"   ✅ GGUF: {gguf_mb:.0f} MB")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create demo TinyLLM model')
    parser.add_argument('--size', default='nano', choices=['micro', 'nano'],
                       help='Model size: micro (100M) or nano (1.5B)')
    parser.add_argument('--q4', action='store_true', help='Also export Q4 GGUF')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    
    if args.size == 'micro':
        hidden, layers, intermediate = 256, 8, 704
        out_dir = args.output_dir or 'demo-tinyllm-micro'
    else:
        hidden, layers, intermediate = 1024, 24, 2816
        out_dir = args.output_dir or 'demo-tinyllm-nano'
    
    model, config = create_demo_model(hidden, layers, 32000, intermediate, 
                                       hidden // 64, out_dir)
    
    # Export GGUF
    gguf_path = out_dir + '.gguf'
    export_to_gguf_demo(model, config, gguf_path)
    
    # Optional Q4
    if args.q4:
        from training.quantize import QuantType
        from export_gguf import export_to_gguf
        export_to_gguf(model, out_dir + '-q4.gguf', config, use_q4=True, qtype=QuantType.Q4_K_M)
        q4_mb = os.path.getsize(out_dir + '-q4.gguf') / (1024**2)
        print(f"   ✅ Q4 GGUF: {q4_mb:.0f} MB")
    
    print(f"\n🎉 Demo model ready!")
    print(f"   HuggingFace: {out_dir}/")
    print(f"   GGUF FP16:   {gguf_path}")
    if args.q4:
        print(f"   GGUF Q4:     {out_dir}-q4.gguf")
    print(f"\n💡 Test with: ./tinyllm run {gguf_path}")
