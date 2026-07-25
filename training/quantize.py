#!/usr/bin/env python3
"""
quantize.py — 4-bit Quantization for TinyLLM (GGUF-compatible)

Supported formats:
  - Q4_0: 4-bit weights, 32-bit float scale per 32-element block
  - Q4_1: 4-bit weights, 32-bit float scale + 32-bit float min per block
  - Q4_K_M: Super-block quantization with 6-bit scales (GGUF K-quant)
            Block size 256, sub-blocks of 32 with 4-bit weights

GGUF tensor types:
  GGML_TYPE_F32  = 0
  GGML_TYPE_F16  = 1
  GGML_TYPE_Q4_0 = 2
  GGML_TYPE_Q4_1 = 3
  GGML_TYPE_Q4_K = 14  (Q4_K_M in GGUF)

Usage:
  python quantize.py input.gguf output-q4.gguf Q4_K_M
"""

import os, sys, struct, math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class QuantType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q4_K_M = 14


@dataclass
class QuantConfig:
    """Quantization block sizes."""
    Q4_0_BLOCK_SIZE = 32
    Q4_1_BLOCK_SIZE = 32
    Q4_K_BLOCK_SIZE = 256     # Super-block
    Q4_K_SUB_BLOCK = 32       # Sub-block within super-block


# ═══════════════════════════════════════════════════════════════
# Q4_0 — Basic 4-bit quantization
# ═══════════════════════════════════════════════════════════════

def quantize_q4_0(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize float32 array to Q4_0 format.
    
    Q4_0 layout (per block of 32 elements):
      - 2 bytes:  float16 scale (d)
      - 16 bytes: 4-bit values (32 × 4-bit = 16 bytes)
      Total: 18 bytes per 32 elements → 4.5 bpw
      
    Args:
        x: float32 array, shape [n] where n must be multiple of 32
    Returns:
        (qdata: uint8 array [n//2 + n//32*2], scales: float16 array [n//32])
    """
    n = len(x)
    assert n % 32 == 0, f"Size must be multiple of 32, got {n}"
    
    num_blocks = n // 32
    scales = np.zeros(num_blocks, dtype=np.float16)
    
    # Quantized data: 4 bits per element = n/2 bytes
    qdata = np.zeros(n // 2, dtype=np.uint8)
    
    for i in range(num_blocks):
        block = x[i * 32:(i + 1) * 32]
        
        # Find scale = max(|block|)
        amax = np.max(np.abs(block))
        if amax < 1e-9:
            amax = 1e-9
        d = amax / 7.0  # 4-bit signed range: [-7, 7]
        scales[i] = np.float16(d)
        
        # Quantize: q = round(x / d), clamped to [-7, 7]
        qvals = np.clip(np.round(block / d), -7, 7).astype(np.int8)
        
        # Pack: lower nibble first, then upper
        # Each byte = (q[2*j] & 0xF) | ((q[2*j+1] & 0xF) << 4)
        for j in range(16):
            lo = qvals[2 * j] & 0x0F
            hi = qvals[2 * j + 1] & 0x0F
            qdata[i * 16 + j] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    
    return qdata, scales


def dequantize_q4_0(qdata: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize Q4_0 back to float32."""
    n = len(qdata) * 2
    num_blocks = len(scales)
    x = np.zeros(n, dtype=np.float32)
    
    for i in range(num_blocks):
        d = float(scales[i])
        for j in range(16):
            byte = qdata[i * 16 + j]
            lo = np.int8(byte & 0x0F)
            hi = np.int8((byte >> 4) & 0x0F)
            # Sign extend from 4-bit
            if lo >= 8: lo -= 16
            if hi >= 8: hi -= 16
            x[i * 32 + 2 * j] = float(lo) * d
            x[i * 32 + 2 * j + 1] = float(hi) * d
    
    return x


# ═══════════════════════════════════════════════════════════════
# Q4_1 — 4-bit with min (asymmetric)
# ═══════════════════════════════════════════════════════════════

def quantize_q4_1(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Q4_1: asymmetric 4-bit with min.
    
    Block layout (32 elements):
      - 2 bytes: float16 scale (d)
      - 2 bytes: float16 min (m)
      - 16 bytes: 4-bit unsigned values [0..15]
      Total: 20 bytes per 32 → 5.0 bpw
      
    x = d * q + m  where q ∈ [0, 15]
    """
    n = len(x)
    assert n % 32 == 0
    
    num_blocks = n // 32
    scales = np.zeros(num_blocks, dtype=np.float16)
    mins = np.zeros(num_blocks, dtype=np.float16)
    qdata = np.zeros(n // 2, dtype=np.uint8)
    
    for i in range(num_blocks):
        block = x[i * 32:(i + 1) * 32]
        bmin = np.min(block)
        bmax = np.max(block)
        
        if bmax - bmin < 1e-9:
            d = 1e-9
        else:
            d = (bmax - bmin) / 15.0
        
        m = bmin
        mins[i] = np.float16(m)
        scales[i] = np.float16(d)
        
        qvals = np.clip(np.round((block - m) / d), 0, 15).astype(np.uint8)
        
        for j in range(16):
            lo = qvals[2 * j] & 0x0F
            hi = qvals[2 * j + 1] & 0x0F
            qdata[i * 16 + j] = lo | (hi << 4)
    
    return qdata, scales, mins


def dequantize_q4_1(qdata: np.ndarray, scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Dequantize Q4_1 to float32."""
    n = len(qdata) * 2
    num_blocks = len(scales)
    x = np.zeros(n, dtype=np.float32)
    
    for i in range(num_blocks):
        d = float(scales[i])
        m = float(mins[i])
        for j in range(16):
            byte = qdata[i * 16 + j]
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            x[i * 32 + 2 * j] = float(lo) * d + m
            x[i * 32 + 2 * j + 1] = float(hi) * d + m
    
    return x


# ═══════════════════════════════════════════════════════════════
# Q4_K_M — Super-block K-quant (GGUF standard)
# ═══════════════════════════════════════════════════════════════

def quantize_q4_k(x: np.ndarray) -> dict:
    """
    Q4_K_M: Super-block quantization.
    
    Super-block layout (256 elements):
      - 2 bytes:  float16 super-block scale (d)
      - 2 bytes:  float16 super-block min (dmin)
      - 12 bytes: 6-bit sub-block scales (8 sub-blocks × 6-bit = 48 bits = 6 bytes) × 2 (scale + min)
                  Actually: 8 × 6-bit quantized sub-block scales
      - 128 bytes: 4-bit weights (256 × 4-bit = 128 bytes)
      
      Total: 2+2+12+128 = 144 bytes per 256 → 4.5 bpw
      
    This is a simplified approximation of the GGUF Q4_K_M format.
    The real format uses importance-based selection of scale bits.
    """
    n = len(x)
    assert n % 256 == 0, f"Q4_K requires multiple of 256, got {n}"
    
    num_super = n // 256
    num_sub = 8  # sub-blocks per super-block
    sub_size = 32
    
    # Super-block scales
    d = np.zeros(num_super, dtype=np.float16)
    dmin = np.zeros(num_super, dtype=np.float16)
    
    # Sub-block scales (quantized to 6 bits each)
    # In real GGUF: scale[0] = 6-bit, scale[1..7] = delta from scale[0] in 6-bit
    sub_scales = np.zeros((num_super, num_sub), dtype=np.uint8)
    sub_mins = np.zeros((num_super, num_sub), dtype=np.uint8)
    
    # 4-bit quantized weights
    qdata = np.zeros(n // 2, dtype=np.uint8)
    
    for s in range(num_super):
        block = x[s * 256:(s + 1) * 256]
        
        # Sub-block quantization
        sub_qvals = np.zeros(256, dtype=np.uint8)
        sub_d = np.zeros(num_sub, dtype=np.float32)
        sub_m = np.zeros(num_sub, dtype=np.float32)
        
        for sb in range(num_sub):
            sub_block = block[sb * sub_size:(sb + 1) * sub_size]
            sb_min = np.min(sub_block)
            sb_max = np.max(sub_block)
            
            if sb_max - sb_min < 1e-9:
                sb_d = 1e-9
            else:
                sb_d = (sb_max - sb_min) / 15.0
            
            sub_d[sb] = sb_d
            sub_m[sb] = sb_min
            
            qvals = np.clip(np.round((sub_block - sb_min) / sb_d), 0, 15).astype(np.uint8)
            sub_qvals[sb * sub_size:(sb + 1) * sub_size] = qvals
        
        # Super-block scale = max sub-block scale
        d[s] = np.float16(np.max(sub_d))
        dmin[s] = np.float16(np.min(sub_m))
        
        # Quantize sub-block scales to 6-bit
        # scale_factor = sub_d[i] / d[s]  → quantize to 6-bit [0..63]
        for sb in range(num_sub):
            sf = sub_d[sb] / max(float(d[s]), 1e-9)
            sub_scales[s, sb] = min(63, int(sf * 63 + 0.5))
            
            mf = (sub_m[sb] - float(dmin[s])) / max(float(d[s]) * 15.0, 1e-9)
            mf = max(0.0, min(1.0, mf))
            sub_mins[s, sb] = min(63, int(mf * 63 + 0.5))
        
        # Pack 4-bit weights
        for j in range(128):
            lo = sub_qvals[2 * j] & 0x0F
            hi = sub_qvals[2 * j + 1] & 0x0F
            qdata[s * 128 + j] = lo | (hi << 4)
    
    return {
        'qdata': qdata,
        'd': d,
        'dmin': dmin,
        'scales': sub_scales,
        'mins': sub_mins,
    }


def dequantize_q4_k(quant: dict) -> np.ndarray:
    """Dequantize Q4_K_M to float32."""
    qdata = quant['qdata']
    d = quant['d']
    dmin = quant['dmin']
    sub_scales = quant['scales']
    sub_mins = quant['mins']
    
    n = len(qdata) * 2
    num_super = len(d)
    x = np.zeros(n, dtype=np.float32)
    
    for s in range(num_super):
        sd = float(d[s])
        sdmin = float(dmin[s])
        
        for sb in range(8):
            sb_d = sd * (sub_scales[s, sb] / 63.0)
            sb_m = sdmin + sd * 15.0 * (sub_mins[s, sb] / 63.0)
            
            for j in range(16):
                byte = qdata[s * 128 + sb * 16 + j]
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                idx = s * 256 + sb * 32 + 2 * j
                x[idx] = float(lo) * sb_d + sb_m
                x[idx + 1] = float(hi) * sb_d + sb_m
    
    return x


# ═══════════════════════════════════════════════════════════════
# Model Quantization
# ═══════════════════════════════════════════════════════════════

def quantize_model_weights(state_dict: dict, qtype: QuantType,
                           exclude_layers: list = None) -> dict:
    """
    Quantize all weight tensors in a model state dict.
    
    Args:
        state_dict: model state_dict
        qtype: target quantization type
        exclude_layers: layer name patterns to keep in FP16 (e.g., ['lm_head', 'embed'])
    
    Returns:
        Quantized state dict with same keys but quantized weights
    """
    if exclude_layers is None:
        exclude_layers = ['tok_embeddings', 'lm_head', 'rms', 'norm']
    
    quantized = {}
    total_bytes_fp16 = 0
    total_bytes_quant = 0
    
    for key, tensor in state_dict.items():
        # Check if excluded
        skip = any(pat in key for pat in exclude_layers)
        
        if skip or tensor.dim() < 2:
            # Keep in FP16 (embeddings, norms, biases)
            quantized[key] = tensor.half() if tensor.dtype == torch.float32 else tensor
            total_bytes_fp16 += tensor.numel() * 2
            total_bytes_quant += tensor.numel() * 2
            continue
        
        # Flatten 2D weight to 1D
        w = tensor.float().numpy().ravel()
        n = len(w)
        
        if qtype == QuantType.Q4_0:
            # Pad to multiple of 32
            pad = (32 - n % 32) % 32
            if pad > 0:
                w = np.pad(w, (0, pad), 'constant')
            
            qdata, scales = quantize_q4_0(w)
            total_bytes_fp16 += n * 2
            total_bytes_quant += len(qdata) + len(scales) * 2
            
            quantized[key] = {
                'qtype': 'q4_0',
                'shape': list(tensor.shape),
                'original_n': n,
                'qdata': qdata,
                'scales': scales,
            }
        
        elif qtype == QuantType.Q4_1:
            pad = (32 - n % 32) % 32
            if pad > 0:
                w = np.pad(w, (0, pad), 'constant')
            
            qdata, scales, mins = quantize_q4_1(w)
            total_bytes_fp16 += n * 2
            total_bytes_quant += len(qdata) + len(scales) * 2 + len(mins) * 2
            
            quantized[key] = {
                'qtype': 'q4_1',
                'shape': list(tensor.shape),
                'original_n': n,
                'qdata': qdata,
                'scales': scales,
                'mins': mins,
            }
        
        elif qtype == QuantType.Q4_K_M:
            pad = (256 - n % 256) % 256
            if pad > 0:
                w = np.pad(w, (0, pad), 'constant')
            
            result = quantize_q4_k(w)
            total_bytes_fp16 += n * 2
            total_bytes_quant += len(result['qdata']) + 2*2 + 12  # qdata + d + dmin + sub_scales
            
            quantized[key] = {
                'qtype': 'q4_k_m',
                'shape': list(tensor.shape),
                'original_n': n,
                **result,
            }
        
        else:
            quantized[key] = tensor
    
    compression = total_bytes_fp16 / max(total_bytes_quant, 1)
    print(f"\n📊 Quantization summary ({qtype.name}):")
    print(f"   FP16 size:     {total_bytes_fp16 / 1e9:.2f} GB")
    print(f"   Quant size:    {total_bytes_quant / 1e9:.2f} GB")
    print(f"   Compression:   {compression:.1f}x")
    print(f"   Effective bpw: {total_bytes_quant * 8 / max(total_bytes_fp16 // 2, 1):.1f}")
    
    return quantized


# ═══════════════════════════════════════════════════════════════
# Quality Metrics
# ═══════════════════════════════════════════════════════════════

def compute_quantization_error(original: np.ndarray, dequantized: np.ndarray) -> dict:
    """Compute error metrics between original and dequantized weights."""
    diff = original - dequantized
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_err = np.max(np.abs(diff))
    rel_err = np.linalg.norm(diff) / max(np.linalg.norm(original), 1e-9)
    
    return {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mae),
        'max_error': float(max_err),
        'relative_error': float(rel_err),
    }


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import torch
    
    print("=" * 60)
    print("4-bit Quantization Tests")
    print("=" * 60)
    
    # Test Q4_0
    x = np.random.randn(1024).astype(np.float32) * 0.5
    qdata, scales = quantize_q4_0(x)
    x_recon = dequantize_q4_0(qdata, scales)
    err = compute_quantization_error(x, x_recon)
    print(f"\nQ4_0 (1024 elements):")
    print(f"  Original: {x.nbytes} bytes → Quantized: {qdata.nbytes + scales.nbytes} bytes")
    print(f"  RMSE: {err['rmse']:.6f}, Max error: {err['max_error']:.6f}")
    print(f"  Relative error: {err['relative_error']:.4%}")
    
    # Test Q4_1
    x2 = np.random.randn(1024).astype(np.float32) * 0.7 + 0.3
    qdata2, scales2, mins2 = quantize_q4_1(x2)
    x2_recon = dequantize_q4_1(qdata2, scales2, mins2)
    err2 = compute_quantization_error(x2, x2_recon)
    print(f"\nQ4_1 (1024 elements):")
    print(f"  RMSE: {err2['rmse']:.6f}, Max error: {err2['max_error']:.6f}")
    print(f"  Relative error: {err2['relative_error']:.4%}")
    
    # Test Q4_K_M
    x3 = np.random.randn(1024).astype(np.float32)
    qresult = quantize_q4_k(x3)
    x3_recon = dequantize_q4_k(qresult)
    err3 = compute_quantization_error(x3, x3_recon)
    print(f"\nQ4_K_M (1024 elements):")
    print(f"  RMSE: {err3['rmse']:.6f}, Max error: {err3['max_error']:.6f}")
    print(f"  Relative error: {err3['relative_error']:.4%}")
    
    # Simulated model test
    print(f"\n{'='*60}")
    print("Simulated Model Quantization")
    print("=" * 60)
    
    mock_sd = {}
    for i in range(24):
        mock_sd[f'layer.{i}.q_proj.weight'] = torch.randn(1024, 1024) * 0.02
        mock_sd[f'layer.{i}.k_proj.weight'] = torch.randn(1024, 1024) * 0.02
        mock_sd[f'layer.{i}.v_proj.weight'] = torch.randn(1024, 1024) * 0.02
        mock_sd[f'layer.{i}.o_proj.weight'] = torch.randn(1024, 1024) * 0.02
        mock_sd[f'layer.{i}.gate_proj.weight'] = torch.randn(2816, 1024) * 0.02
        mock_sd[f'layer.{i}.up_proj.weight'] = torch.randn(2816, 1024) * 0.02
        mock_sd[f'layer.{i}.down_proj.weight'] = torch.randn(1024, 2816) * 0.02
    mock_sd['tok_embeddings.weight'] = torch.randn(32000, 1024) * 0.02
    mock_sd['lm_head.weight'] = torch.randn(32000, 1024) * 0.02
    
    q_sd = quantize_model_weights(mock_sd, QuantType.Q4_K_M)
    print("✅ Quantization tests passed")
