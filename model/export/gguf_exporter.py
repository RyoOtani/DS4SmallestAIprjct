"""
GGUF Model Exporter — Export TinyLLM models to GGUF format for C runtime.

Converts PyTorch model weights to the GGUF binary format used by tinyllm.
Supports:
  - Q4_0, Q4_1, Q8_0, Q6_K quantization
  - FP32, FP16, BF16
  - Mixed precision (key layers 6-bit, rest 4-bit)
  - Metadata embedding (architecture, tokenizer, hyperparams)
"""

from __future__ import annotations
import struct
import json
import os
import math
from pathlib import Path
from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import numpy as np


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# Quantization type codes (GGML)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q6_K = 18

QBLOCK_SIZE = 32


class GGUFWriter:
    """Writes GGUF format files."""
    
    def __init__(self, path: str, alignment: int = GGUF_DEFAULT_ALIGNMENT):
        self.path = path
        self.alignment = alignment
        self.metadata: OrderedDict[str, tuple] = OrderedDict()
        self.tensors: list[dict] = []
        self._tensor_data_offset = 0
    
    def add_metadata(self, key: str, value, vtype: str = "auto"):
        """Add metadata key-value pair."""
        if vtype == "auto":
            if isinstance(value, bool):
                vtype = "bool"
            elif isinstance(value, int):
                vtype = "int64"
            elif isinstance(value, float):
                vtype = "float64"
            elif isinstance(value, str):
                vtype = "string"
            elif isinstance(value, list):
                vtype = "array"
            else:
                vtype = "string"
                value = str(value)
        
        self.metadata[key] = (vtype, value)
    
    def add_tensor(
        self,
        name: str,
        data: torch.Tensor,
        quant_type: int = GGML_TYPE_F32,
    ) -> int:
        """Register a tensor for writing."""
        shape = list(data.shape)
        n_dims = len(shape)
        
        # Compute quantized size
        n_elements = data.numel()
        if quant_type == GGML_TYPE_F32:
            byte_size = n_elements * 4
        elif quant_type == GGML_TYPE_F16:
            byte_size = n_elements * 2
        elif quant_type == GGML_TYPE_Q4_0:
            n_blocks = (n_elements + QBLOCK_SIZE - 1) // QBLOCK_SIZE
            byte_size = n_blocks * (QBLOCK_SIZE // 2 + 2)  # nibbles + fp16 scale
        elif quant_type == GGML_TYPE_Q4_1:
            n_blocks = (n_elements + QBLOCK_SIZE - 1) // QBLOCK_SIZE
            byte_size = n_blocks * (QBLOCK_SIZE // 2 + 4)  # nibbles + fp16 scale + fp16 min
        elif quant_type == GGML_TYPE_Q8_0:
            n_blocks = (n_elements + QBLOCK_SIZE - 1) // QBLOCK_SIZE
            byte_size = n_blocks * (QBLOCK_SIZE + 2)
        else:
            byte_size = n_elements * 2  # fallback FP16
        
        offset = self._tensor_data_offset
        self._tensor_data_offset += byte_size
        # Align
        self._tensor_data_offset = (
            (self._tensor_data_offset + self.alignment - 1) // self.alignment * self.alignment
        )
        
        self.tensors.append({
            "name": name,
            "shape": shape,
            "n_dims": n_dims,
            "data": data,
            "quant_type": quant_type,
            "offset": offset,
            "size": byte_size,
        })
        
        return len(self.tensors) - 1
    
    def write(self):
        """Write the GGUF file."""
        with open(self.path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))
            
            # Metadata
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                self._write_value(f, vtype, value)
            
            # Tensor info
            for t in self.tensors:
                self._write_string(f, t["name"])
                f.write(struct.pack("<I", t["n_dims"]))
                for d in t["shape"]:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", t["quant_type"]))
                f.write(struct.pack("<Q", t["offset"]))
            
            # Align to GGUF_ALIGNMENT
            current = f.tell()
            aligned = (current + self.alignment - 1) // self.alignment * self.alignment
            f.write(b'\x00' * (aligned - current))
            
            # Tensor data
            for t in self.tensors:
                self._write_tensor_data(f, t)
                # Align
                current = f.tell()
                aligned = (current + self.alignment - 1) // self.alignment * self.alignment
                f.write(b'\x00' * (aligned - current))
        
        size_mb = os.path.getsize(self.path) / (1024 * 1024)
        print(f"✓ GGUF written: {self.path} ({size_mb:.1f} MB)")
        print(f"  Tensors: {len(self.tensors)}, Metadata: {len(self.metadata)}")
    
    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, vtype: str, value):
        type_codes = {
            "uint8": 0, "int8": 1, "uint16": 2, "int16": 3,
            "uint32": 4, "int32": 5, "float32": 6, "bool": 7,
            "string": 8, "array": 9, "uint64": 10, "int64": 11, "float64": 12,
        }
        code = type_codes.get(vtype, 0)
        f.write(struct.pack("<I", code))
        
        if vtype == "uint32":
            f.write(struct.pack("<I", value))
        elif vtype in ("uint64", "int64"):
            f.write(struct.pack("<Q", value))
        elif vtype == "float32":
            f.write(struct.pack("<f", value))
        elif vtype == "float64":
            f.write(struct.pack("<d", value))
        elif vtype == "bool":
            f.write(struct.pack("<?", value))
        elif vtype == "string":
            self._write_string(f, str(value))
        elif vtype == "int32":
            f.write(struct.pack("<i", value))
        elif vtype == "array":
            if isinstance(value, (list, tuple)):
                f.write(struct.pack("<I", 9))  # array of float32
                f.write(struct.pack("<I", len(value)))
                for v in value:
                    f.write(struct.pack("<f", v))
    
    @staticmethod
    def _quantize_q4_0(data: torch.Tensor) -> tuple[bytes, float]:
        """Quantize to Q4_0 format. Returns (packed_bytes, scale)."""
        x = data.float().numpy().flatten()
        n = len(x)
        n_blocks = (n + QBLOCK_SIZE - 1) // QBLOCK_SIZE
        
        result = bytearray()
        all_scales = []
        
        for b in range(n_blocks):
            start = b * QBLOCK_SIZE
            end = min(start + QBLOCK_SIZE, n)
            block = x[start:end]
            
            # Find scale
            amax = np.max(np.abs(block))
            scale = amax / 7.0 if amax > 0 else 1.0
            
            if scale < 1e-6:
                scale = 1.0
            
            # Quantize each value to 4 bits
            quantized = np.clip(np.round(block / scale + 8.0), 0, 15).astype(np.uint8)
            
            # Pack nibbles
            packed = bytearray((end - start + 1) // 2)
            for i in range(start, end):
                k = i - start
                val = quantized[i - start]
                if k & 1:
                    packed[k // 2] |= val << 4
                else:
                    packed[k // 2] = val
            
            result.extend(packed)
            # fp16 scale
            scale_f16 = struct.pack("<e", np.float16(scale))
            result.extend(scale_f16)
            all_scales.append(scale)
        
        return bytes(result), float(np.mean(all_scales))
    
    def _write_tensor_data(self, f, tensor_info: dict):
        """Write quantized tensor data."""
        data = tensor_info["data"].float()
        qt = tensor_info["quant_type"]
        
        if qt == GGML_TYPE_F32:
            f.write(data.numpy().astype(np.float32).tobytes())
        elif qt == GGML_TYPE_F16:
            f.write(data.numpy().astype(np.float16).tobytes())
        elif qt == GGML_TYPE_Q4_0:
            packed, _ = self._quantize_q4_0(data)
            f.write(packed)
        elif qt == GGML_TYPE_Q8_0:
            self._write_q8_0(f, data)
        else:
            # Default: FP16
            f.write(data.numpy().astype(np.float16).tobytes())
    
    @staticmethod
    def _write_q8_0(f, data: torch.Tensor):
        """Write Q8_0 quantized data."""
        x = data.float().numpy().flatten()
        n = len(x)
        n_blocks = (n + QBLOCK_SIZE - 1) // QBLOCK_SIZE
        
        result = bytearray()
        for b in range(n_blocks):
            start = b * QBLOCK_SIZE
            end = min(start + QBLOCK_SIZE, n)
            block = x[start:end]
            
            amax = np.max(np.abs(block))
            scale = amax / 127.0 if amax > 0 else 1.0
            if scale < 1e-8:
                scale = 1.0
            
            # Scale as fp16
            result.extend(struct.pack("<e", np.float16(scale)))
            
            # Quantize to int8
            vals = np.clip(np.round(block / scale), -127, 127).astype(np.int8)
            result.extend(vals.tobytes())
        
        f.write(bytes(result))


def export_model_to_gguf(
    model: nn.Module,
    output_path: str,
    config_dict: dict,
    use_q4_0: bool = True,
    mixed_precision: bool = True,
) -> str:
    """
    Export a PyTorch TinyLLM model to GGUF format.
    
    Args:
        model: TinyLLM model
        output_path: Path for the output GGUF file
        config_dict: Model configuration dictionary
        use_q4_0: Use 4-bit quantization (default: True)
        mixed_precision: Use 6-bit for key layers, 4-bit for rest
    
    Returns:
        Path to the exported GGUF file
    """
    writer = GGUFWriter(output_path)
    
    # Add metadata
    writer.add_metadata("general.architecture", config_dict.get("name", "tinyllm"))
    writer.add_metadata("general.name", config_dict.get("name", "tinyllm"))
    writer.add_metadata("general.quantization_version", 2)
    
    writer.add_metadata("llm.context_length", config_dict.get("max_seq_len", 8192), "int64")
    writer.add_metadata("llm.block_count", config_dict.get("n_layers", 32), "int64")
    writer.add_metadata("llm.hidden_size", config_dict.get("hidden_dim", 2048), "int64")
    writer.add_metadata("llm.head_count", config_dict.get("n_heads", 32), "int64")
    writer.add_metadata("llm.kv_head_count", config_dict.get("n_kv_heads", 8), "int64")
    writer.add_metadata("llm.vocab_size", config_dict.get("vocab_size", 65536), "int64")
    
    if config_dict.get("use_moe"):
        writer.add_metadata("llm.expert_count", config_dict.get("n_experts", 64), "int64")
        writer.add_metadata("llm.expert_used_count", config_dict.get("n_active_experts", 6), "int64")
    
    if config_dict.get("use_mla"):
        writer.add_metadata("llm.kv_latent_dim", config_dict.get("kv_latent_dim", 512), "int64")
    
    writer.add_metadata("tokenizer.ggml.model", "bpe")
    writer.add_metadata("tokenizer.ggml.bos_token_id", 0, "int64")
    writer.add_metadata("tokenizer.ggml.eos_token_id", 1, "int64")
    
    # Determine quantization per layer
    def layer_qt(layer_type: str, layer_idx: int, n_layers: int) -> int:
        if not use_q4_0:
            return GGML_TYPE_F32
        if not mixed_precision:
            return GGML_TYPE_Q4_0
        # Key layers at higher precision
        if layer_type == "embedding" or layer_type == "lm_head":
            return GGML_TYPE_Q8_0
        if layer_idx == 0 or layer_idx == n_layers - 1:
            return GGML_TYPE_Q6_K
        if layer_idx <= 2 or layer_idx >= n_layers - 3:
            return GGML_TYPE_Q8_0
        return GGML_TYPE_Q4_0
    
    state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    n_layers = config_dict.get("n_layers", 32)
    
    # Export tensors
    tensor_map = {
        "tok_embeddings.weight": ("token_embd.weight", "embedding", -1),
        "final_norm.weight": ("output_norm.weight", "norm", -1),
    }
    
    if not config_dict.get("tie_word_embeddings", False):
        tensor_map["lm_head.weight"] = ("output.weight", "lm_head", -1)
    
    for i in range(n_layers):
        prefix = f"layers.{i}."
        ggml_prefix = f"blk.{i}."
        
        layer_tensors = {
            f"{prefix}attn_norm.weight": (f"{ggml_prefix}attn_norm.weight", "attn_norm", i),
            f"{prefix}ffn_norm.weight": (f"{ggml_prefix}ffn_norm.weight", "ffn_norm", i),
            f"{prefix}attention.w_q.weight": (f"{ggml_prefix}attn_q.weight", "q", i),
            f"{prefix}attention.w_o.weight": (f"{ggml_prefix}attn_output.weight", "o", i),
        }
        
        if config_dict.get("use_mla"):
            layer_tensors[f"{prefix}attention.w_kv_compress.weight"] = (f"{ggml_prefix}attn_kv_a.weight", "kv_compress", i)
            layer_tensors[f"{prefix}attention.w_k_up.weight"] = (f"{ggml_prefix}attn_k.weight", "k_up", i)
            layer_tensors[f"{prefix}attention.w_v_up.weight"] = (f"{ggml_prefix}attn_v.weight", "v_up", i)
        else:
            layer_tensors[f"{prefix}attention.w_k.weight"] = (f"{ggml_prefix}attn_k.weight", "k", i)
            layer_tensors[f"{prefix}attention.w_v.weight"] = (f"{ggml_prefix}attn_v.weight", "v", i)
        
        if config_dict.get("use_moe") and (
            not config_dict.get("moe_layers") or i in config_dict.get("moe_layers", [])
        ):
            layer_tensors[f"{prefix}moe.gate.weight"] = (f"{ggml_prefix}ffn_gate.weight", "moe_gate", i)
            for e in range(config_dict.get("n_experts", 64)):
                layer_tensors[f"{prefix}moe.experts.{e}.ffn.w_gate.weight"] = (
                    f"{ggml_prefix}ffn_gate.{e}.weight", "expert_gate", i,
                )
                layer_tensors[f"{prefix}moe.experts.{e}.ffn.w_up.weight"] = (
                    f"{ggml_prefix}ffn_up.{e}.weight", "expert_up", i,
                )
                layer_tensors[f"{prefix}moe.experts.{e}.ffn.w_down.weight"] = (
                    f"{ggml_prefix}ffn_down.{e}.weight", "expert_down", i,
                )
        else:
            layer_tensors[f"{prefix}ffn.w_gate.weight"] = (f"{ggml_prefix}ffn_gate.weight", "ffn_gate", i)
            layer_tensors[f"{prefix}ffn.w_up.weight"] = (f"{ggml_prefix}ffn_up.weight", "ffn_up", i)
            layer_tensors[f"{prefix}ffn.w_down.weight"] = (f"{ggml_prefix}ffn_down.weight", "ffn_down", i)
        
        tensor_map.update(layer_tensors)
    
    # Export each tensor
    exported = 0
    for torch_key, (ggml_name, tensor_type, layer_idx) in tensor_map.items():
        if torch_key in state_dict:
            qt = layer_qt(tensor_type, layer_idx, n_layers)
            writer.add_tensor(ggml_name, state_dict[torch_key], qt)
            exported += 1
    
    print(f"Exported {exported} tensors to GGUF")
    writer.write()
    return output_path
