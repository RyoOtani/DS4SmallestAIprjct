#!/usr/bin/env python3
"""
export_gguf.py — Convert notebook-trained TinyLLM-nano to GGUF for C engine.

Maps the simplified notebook model (standard attention, dense FFN) to the
GGUF format that the tinyllm C runtime expects (MLA + dense FFN).

Strategy:
  - w_kv_compress = Identity matrix (MLA compression becomes no-op)
  - kv_latent_dim = hidden_dim (no actual compression)
  - k_proj → w_k_up, v_proj → w_v_up (same shape when latent==hidden)

Usage:
  python export_gguf.py [--fp32] [--output model.gguf]
"""

import os, sys, json, struct, time
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════
# GGUF constants (matching src/model.c and include/config.h)
# ═══════════════════════════════════════════════════════════════════

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2

# GGUF value types
GGUF_TYPE_U32 = 4
GGUF_TYPE_I32 = 5
GGUF_TYPE_F32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STR = 8
GGUF_TYPE_U64 = 10
GGUF_TYPE_I64 = 11
GGUF_TYPE_F64 = 12


def write_string(f, s: str):
    """Write GGUF string (u64 length + data)."""
    data = s.encode('utf-8')
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def write_value(f, vtype: int, value):
    """Write typed value (type tag + value)."""
    f.write(struct.pack('<I', vtype))  # ← type tag first!
    if vtype == GGUF_TYPE_U32:
        f.write(struct.pack('<I', int(value)))
    elif vtype == GGUF_TYPE_I32:
        f.write(struct.pack('<i', int(value)))
    elif vtype == GGUF_TYPE_F32:
        f.write(struct.pack('<f', float(value)))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(struct.pack('<?', bool(value)))
    elif vtype == GGUF_TYPE_STR:
        write_string(f, str(value))
    elif vtype == GGUF_TYPE_U64:
        f.write(struct.pack('<Q', int(value)))
    elif vtype == GGUF_TYPE_I64:
        f.write(struct.pack('<q', int(value)))
    elif vtype == GGUF_TYPE_F64:
        f.write(struct.pack('<d', float(value)))
    else:
        raise ValueError(f"Unknown value type: {vtype}")


def align_file(f, alignment=GGUF_ALIGNMENT):
    """Pad to alignment."""
    pos = f.tell()
    aligned = (pos + alignment - 1) // alignment * alignment
    f.write(b'\x00' * (aligned - pos))


# ═══════════════════════════════════════════════════════════════════
# Model class (must match the notebook's Cell 11)
# ═══════════════════════════════════════════════════════════════════

class TinyLLMLayer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg['hidden_size']
        inter = cfg.get('intermediate_size', D * 11 // 4)
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
        raise NotImplementedError("Export only")


class TinyLLMModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = torch.nn.Embedding(cfg['vocab_size'], cfg['hidden_size'])
        self.layers = torch.nn.ModuleList([
            TinyLLMLayer(cfg) for _ in range(cfg['num_hidden_layers'])
        ])
        self.norm = torch.nn.RMSNorm(cfg['hidden_size'], eps=1e-5)
        self.lm_head = torch.nn.Linear(cfg['hidden_size'], cfg['vocab_size'], bias=False)

    def forward(self, x):
        raise NotImplementedError("Export only")


# ═══════════════════════════════════════════════════════════════════
# GGUF Export
# ═══════════════════════════════════════════════════════════════════

def export_to_gguf(model, output_path, config, use_q4=False):
    """Export notebook TinyLLM model to GGUF for C engine."""
    D = config['hidden_size']
    V = config['vocab_size']
    L = config['num_hidden_layers']
    n_heads = 16
    head_dim = D // n_heads

    # We use kv_latent = D so that w_kv_compress = I is a no-op
    kv_latent = D

    state_dict = model.state_dict()
    tensors = []  # list of (name, data, ggml_type, shape)

    def add_tensor(name, data, qtype=GGML_TYPE_F32):
        data_np = data.detach().cpu().float().numpy()
        tensors.append((name, data_np, qtype))

    # ── Embedding ───────────────────────────────────────────────
    add_tensor("token_embd.weight", state_dict["embed.weight"])

    # ── Layers ──────────────────────────────────────────────────
    for l in range(L):
        p = f"layers.{l}."

        # Attention: notebook → GGUF mapping
        # q_proj (D×D) → attn_q (D×D) — OK
        add_tensor(f"blk.{l}.attn_q.weight", state_dict[f"{p}q_proj.weight"])

        # k_proj (D×D) → attn_k (D×D) — when kv_latent==D, w_k_up is [n_heads*head_dim, latent] = [D, D]
        add_tensor(f"blk.{l}.attn_k.weight", state_dict[f"{p}k_proj.weight"])

        # v_proj (D×D) → attn_v (D×D) — same reasoning
        add_tensor(f"blk.{l}.attn_v.weight", state_dict[f"{p}v_proj.weight"])

        # o_proj → attn_output
        add_tensor(f"blk.{l}.attn_output.weight", state_dict[f"{p}o_proj.weight"])

        # kV compress = Identity (so no actual compression happens)
        identity = np.eye(D, dtype=np.float32)
        tensors.append((f"blk.{l}.attn_kv_a.weight", identity, GGML_TYPE_F32))

        # RMS Norms
        add_tensor(f"blk.{l}.attn_norm.weight", state_dict[f"{p}norm1.weight"])
        add_tensor(f"blk.{l}.ffn_norm.weight", state_dict[f"{p}norm2.weight"])

        # FFN (dense)
        add_tensor(f"blk.{l}.ffn_gate.weight", state_dict[f"{p}gate_proj.weight"])
        add_tensor(f"blk.{l}.ffn_up.weight", state_dict[f"{p}up_proj.weight"])
        add_tensor(f"blk.{l}.ffn_down.weight", state_dict[f"{p}down_proj.weight"])

    # ── Final norm ──────────────────────────────────────────────
    add_tensor("output_norm.weight", state_dict["norm.weight"])

    # ── LM head ─────────────────────────────────────────────────
    add_tensor("output.weight", state_dict["lm_head.weight"])

    # ── Write GGUF file ─────────────────────────────────────────
    n_tensors = len(tensors)
    n_meta = 20  # pre-count

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_meta))

        # Metadata
        metadata = [
            ("general.architecture", GGUF_TYPE_STR, "tinyllm-nano"),
            ("general.name", GGUF_TYPE_STR, "tinyllm-nano"),
            ("llm.context_length", GGUF_TYPE_U64, config.get('max_position_embeddings', 8192)),
            ("llm.block_count", GGUF_TYPE_U64, L),
            ("llm.hidden_size", GGUF_TYPE_U64, D),
            ("llm.head_count", GGUF_TYPE_U64, n_heads),
            ("llm.head_dim", GGUF_TYPE_U64, head_dim),
            ("llm.kv_head_count", GGUF_TYPE_U64, n_heads),
            ("llm.vocab_size", GGUF_TYPE_U64, V),
            ("llm.kv_latent_dim", GGUF_TYPE_U64, kv_latent),
            ("llm.rope_theta", GGUF_TYPE_F32, 10000.0),
            ("llm.expert_count", GGUF_TYPE_U64, 0),
            ("llm.expert_used_count", GGUF_TYPE_U64, 0),
            ("llm.feed_forward_length", GGUF_TYPE_U64, config.get('intermediate_size', 2816)),
            ("tokenizer.ggml.model", GGUF_TYPE_STR, "bpe"),
            ("tokenizer.ggml.bos_token_id", GGUF_TYPE_U64, 0),
            ("tokenizer.ggml.eos_token_id", GGUF_TYPE_U64, 1),
            ("tokenizer.ggml.pad_token_id", GGUF_TYPE_U64, 2),
            ("general.quantization_version", GGUF_TYPE_U32, 2),
            ("general.file_type", GGUF_TYPE_U32, 1 if use_q4 else 0),
        ]

        for key, vtype, value in metadata:
            write_string(f, key)
            write_value(f, vtype, value)

        # Tensor info
        offset = 0
        for name, data, qtype in tensors:
            write_string(f, name)
            shape = data.shape
            n_dims = len(shape)
            f.write(struct.pack('<I', n_dims))
            for d in shape:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', qtype))

            # Compute byte size
            nelem = int(np.prod(shape))
            if qtype == GGML_TYPE_F32:
                bsize = nelem * 4
            elif qtype == GGML_TYPE_F16:
                bsize = nelem * 2
            elif qtype == GGML_TYPE_Q4_0:
                n_blocks = (nelem + 31) // 32
                bsize = n_blocks * (16 + 2)  # nibbles + fp16 scale
            else:
                bsize = nelem * 4

            f.write(struct.pack('<Q', offset))
            offset += bsize
            offset = (offset + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT

        # Align
        align_file(f)

        # Tensor data
        for name, data, qtype in tensors:
            if qtype == GGML_TYPE_F32:
                f.write(data.astype(np.float32).tobytes())
            elif qtype == GGML_TYPE_F16:
                f.write(data.astype(np.float16).tobytes())
            else:
                f.write(data.astype(np.float32).tobytes())
            align_file(f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported: {output_path} ({size_mb:.0f} MB, {n_tensors} tensors)")
    print(f"   dim={D}, layers={L}, vocab={V}, heads={n_heads}")
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    MODEL_DIR = "downloaded_models/tinyllm-nano"
    OUTPUT = "tinyllm-nano.gguf"

    # Parse args
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        OUTPUT = sys.argv[idx + 1]
    use_q4 = '--q4' in sys.argv

    # Load config
    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)
    config['vocab_size'] = 4639
    config['hidden_size'] = 1024
    config['num_hidden_layers'] = 24
    config['intermediate_size'] = 2816

    print(f"📋 Config: hidden={config['hidden_size']}, layers={config['num_hidden_layers']}, vocab={config['vocab_size']}")

    # Create model
    model = TinyLLMModel(config)

    # Load weights
    ckpt = torch.load(f"{MODEL_DIR}/model.pt", map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"✅ Loaded checkpoint ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # Export
    export_to_gguf(model, OUTPUT, config, use_q4=use_q4)
    print(f"\n🎉 Ready! Run: ./tinyllm run {OUTPUT}")


if __name__ == "__main__":
    main()
