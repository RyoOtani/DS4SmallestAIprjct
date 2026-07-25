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

# Import quantization module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.quantize import (
    QuantType, quantize_q4_0, quantize_q4_1, quantize_q4_k,
    dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    quantize_model_weights,
)

# ═══════════════════════════════════════════════════════════════════
# GGUF constants (matching src/model.c and include/config.h)
# ═══════════════════════════════════════════════════════════════════

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q4_K = 14  # Q4_K_M in GGUF

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

def export_to_gguf(model, output_path, config, use_q4=False, qtype=QuantType.Q4_K_M):
    """Export notebook TinyLLM model to GGUF for C engine.
    
    Args:
        model: TinyLLMModel instance
        output_path: output .gguf file path
        config: model config dict
        use_q4: if True, quantize weights to 4-bit
        qtype: quantization type (Q4_0, Q4_1, Q4_K_M)
    """
    D = config['hidden_size']
    V = config['vocab_size']
    L = config['num_hidden_layers']
    n_heads = 16
    head_dim = D // n_heads

    # We use kv_latent = D so that w_kv_compress = I is a no-op
    kv_latent = D

    state_dict = model.state_dict()
    tensors = []  # list of (name, data_or_dict, ggml_type, shape)

    def add_tensor(name, data, qtype_override=None):
        """Add tensor, optionally quantizing if use_q4 is True."""
        data_np = data.detach().cpu().float().numpy()
        
        if use_q4 and data_np.ndim >= 2:
            # Quantize weight matrices (not norms, embeddings, etc.)
            n = data_np.size
            if qtype == QuantType.Q4_0:
                pad = (32 - n % 32) % 32
                if pad > 0:
                    data_np = np.pad(data_np.ravel(), (0, pad), 'constant').reshape(-1)
                else:
                    data_np = data_np.ravel()
                qdata, scales = quantize_q4_0(data_np)
                tensors.append((name, {'qdata': qdata, 'scales': scales, 'qtype': 'q4_0'},
                               GGML_TYPE_Q4_0, list(data.shape)))
            elif qtype == QuantType.Q4_1:
                pad = (32 - n % 32) % 32
                if pad > 0:
                    data_np = np.pad(data_np.ravel(), (0, pad), 'constant').reshape(-1)
                else:
                    data_np = data_np.ravel()
                qdata, scales, mins = quantize_q4_1(data_np)
                tensors.append((name, {'qdata': qdata, 'scales': scales, 'mins': mins, 'qtype': 'q4_1'},
                               GGML_TYPE_Q4_1, list(data.shape)))
            elif qtype == QuantType.Q4_K_M:
                pad = (256 - n % 256) % 256
                if pad > 0:
                    data_np = np.pad(data_np.ravel(), (0, pad), 'constant').reshape(-1)
                else:
                    data_np = data_np.ravel()
                result = quantize_q4_k(data_np)
                result['qtype'] = 'q4_k_m'
                tensors.append((name, result, GGML_TYPE_Q4_K, list(data.shape)))
            else:
                data_np = data.detach().cpu().float().numpy()
                tensors.append((name, data_np, qtype_override or GGML_TYPE_F32, list(data_np.shape)))
        else:
            qt = qtype_override or GGML_TYPE_F32
            tensors.append((name, data_np, qt, list(data_np.shape)))

    # ── Embedding (keep FP32 — small, needs precision) ─────────
    tensors.append(("token_embd.weight",
                    state_dict["embed.weight"].detach().cpu().float().numpy(),
                    GGML_TYPE_F32,
                    list(state_dict["embed.weight"].shape)))

    # ── Layers ──────────────────────────────────────────────────
    for l in range(L):
        p = f"layers.{l}."
        add_tensor(f"blk.{l}.attn_q.weight", state_dict[f"{p}q_proj.weight"])
        add_tensor(f"blk.{l}.attn_k.weight", state_dict[f"{p}k_proj.weight"])
        add_tensor(f"blk.{l}.attn_v.weight", state_dict[f"{p}v_proj.weight"])
        add_tensor(f"blk.{l}.attn_output.weight", state_dict[f"{p}o_proj.weight"])
        
        # KV compress = Identity (no actual compression, needs to be FP32)
        identity = np.eye(D, dtype=np.float32)
        tensors.append((f"blk.{l}.attn_kv_a.weight", identity, GGML_TYPE_F32, [D, D]))
        
        # RMS Norms (keep FP32 — small)
        tensors.append((f"blk.{l}.attn_norm.weight",
                        state_dict[f"{p}norm1.weight"].detach().cpu().float().numpy(),
                        GGML_TYPE_F32, list(state_dict[f"{p}norm1.weight"].shape)))
        tensors.append((f"blk.{l}.ffn_norm.weight",
                        state_dict[f"{p}norm2.weight"].detach().cpu().float().numpy(),
                        GGML_TYPE_F32, list(state_dict[f"{p}norm2.weight"].shape)))
        
        # FFN
        add_tensor(f"blk.{l}.ffn_gate.weight", state_dict[f"{p}gate_proj.weight"])
        add_tensor(f"blk.{l}.ffn_up.weight", state_dict[f"{p}up_proj.weight"])
        add_tensor(f"blk.{l}.ffn_down.weight", state_dict[f"{p}down_proj.weight"])

    # ── Final norm ──────────────────────────────────────────────
    tensors.append(("output_norm.weight",
                    state_dict["norm.weight"].detach().cpu().float().numpy(),
                    GGML_TYPE_F32, list(state_dict["norm.weight"].shape)))

    # ── LM head (keep FP32 — critical for output quality) ──────
    tensors.append(("output.weight",
                    state_dict["lm_head.weight"].detach().cpu().float().numpy(),
                    GGML_TYPE_F32, list(state_dict["lm_head.weight"].shape)))

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
        for name, data, qtype, shape in tensors:
            write_string(f, name)
            n_dims = len(shape)
            f.write(struct.pack('<I', n_dims))
            for d in shape:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', qtype))

            # Compute byte size based on type
            if qtype == GGML_TYPE_F32:
                bsize = int(np.prod(shape)) * 4
            elif qtype == GGML_TYPE_F16:
                bsize = int(np.prod(shape)) * 2
            elif qtype in (GGML_TYPE_Q4_0, GGML_TYPE_Q4_1):
                # Block size 32: 16 bytes nibbles + 2 bytes scale (+2 bytes min for Q4_1)
                nelem = int(np.prod(shape))
                n_blocks = (nelem + 31) // 32
                bsize = n_blocks * (16 + 2)
                if qtype == GGML_TYPE_Q4_1:
                    bsize += n_blocks * 2  # extra min per block
            elif qtype == GGML_TYPE_Q4_K:
                # Super-block 256: 128 bytes nibbles + 2+2+12 bytes scale data
                nelem = int(np.prod(shape))
                n_super = (nelem + 255) // 256
                bsize = n_super * (128 + 2 + 2 + 12)
            else:
                bsize = int(np.prod(shape)) * 4

            f.write(struct.pack('<Q', offset))
            offset += bsize
            offset = (offset + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT

        # Align
        align_file(f)

        # Tensor data
        for name, data, qtype, shape in tensors:
            if qtype == GGML_TYPE_F32:
                f.write(data.astype(np.float32).tobytes())
            elif qtype == GGML_TYPE_F16:
                f.write(data.astype(np.float16).tobytes())
            elif qtype == GGML_TYPE_Q4_0:
                # Write Q4_0: qdata (nibbles) + scales (fp16)
                f.write(data['qdata'].tobytes())
                f.write(data['scales'].astype(np.float16).tobytes())
            elif qtype == GGML_TYPE_Q4_1:
                # Write Q4_1: qdata + scales + mins
                f.write(data['qdata'].tobytes())
                f.write(data['scales'].astype(np.float16).tobytes())
                f.write(data['mins'].astype(np.float16).tobytes())
            elif qtype == GGML_TYPE_Q4_K:
                # Write Q4_K_M: qdata + d + dmin + sub_scales + sub_mins
                f.write(data['qdata'].tobytes())
                f.write(data['d'].astype(np.float16).tobytes())
                f.write(data['dmin'].astype(np.float16).tobytes())
                f.write(data['scales'].tobytes())
                f.write(data['mins'].tobytes())
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
    QTYPE = QuantType.F32

    # Parse args
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        OUTPUT = sys.argv[idx + 1]
    
    use_q4 = '--q4' in sys.argv or '--q4_0' in sys.argv or '--q4_k' in sys.argv
    
    if '--q4_k' in sys.argv:
        QTYPE = QuantType.Q4_K_M
    elif '--q4_1' in sys.argv:
        QTYPE = QuantType.Q4_1
    elif '--q4' in sys.argv or '--q4_0' in sys.argv:
        QTYPE = QuantType.Q4_0
    
    # Load config
    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)
    config['vocab_size'] = 32000  # v7 tokenizer
    config['hidden_size'] = 1024
    config['num_hidden_layers'] = 24
    config['intermediate_size'] = 2816

    print(f"📋 Config: hidden={config['hidden_size']}, layers={config['num_hidden_layers']}, vocab={config['vocab_size']}")
    if use_q4:
        print(f"🔧 Quantization: {QTYPE.name}")

    # Create model
    model = TinyLLMModel(config)

    # Load weights
    ckpt = torch.load(f"{MODEL_DIR}/model.pt", map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded checkpoint ({total_params/1e6:.0f}M params)")
    
    # Estimated sizes
    fp32_mb = total_params * 4 / (1024**2)
    fp16_mb = total_params * 2 / (1024**2)
    q4_mb = total_params * 0.55  # ~4.5 bpw ≈ 0.55 bytes per param
    print(f"   Est. FP32: {fp32_mb:.0f} MB | FP16: {fp16_mb:.0f} MB | Q4: {q4_mb:.0f} MB")

    # Export
    export_to_gguf(model, OUTPUT, config, use_q4=use_q4, qtype=QTYPE)
    actual_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\n🎉 Ready! {actual_mb:.0f} MB → ./tinyllm run {OUTPUT}")


if __name__ == "__main__":
    main()
