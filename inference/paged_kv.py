#!/usr/bin/env python3
"""
paged_kv.py — Paged KV Cache for TinyLLM

vLLM-style PagedAttention: KV cache stored in fixed-size blocks,
managed via a block table. Enables:
  - Zero memory fragmentation
  - Memory sharing across sequences (beam search, prefix caching)
  - Dynamic memory allocation/deallocation
  - Efficient continuous batching

Block layout:
  Each block stores KV for [num_layers][block_size][num_heads][head_dim]
  
  Physical memory: [num_blocks][block_size][num_kv_heads][head_dim]
  Block table:     [batch_size][max_blocks_per_seq] → physical block indices
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import ctypes


@dataclass
class PagedKVConfig:
    """Configuration for paged KV cache."""
    num_layers: int = 24
    num_heads: int = 16           # Q heads
    num_kv_heads: int = 4         # KV heads (GQA)
    head_dim: int = 64            # D // n_heads
    block_size: int = 16          # Tokens per block
    max_num_blocks: int = 1024    # Total physical blocks
    dtype: torch.dtype = torch.float16


class BlockTable:
    """
    Manages mapping from logical blocks (per sequence) to physical blocks.
    
    Logical view:  sequence_i → [block_0, block_1, ..., block_n]
    Physical view: block_j stored at physical_block[j] in memory pool
    
    Free blocks are tracked via a free list for O(1) allocation.
    """
    
    def __init__(self, max_num_blocks: int):
        self.max = max_num_blocks
        self.free_blocks = list(range(max_num_blocks))  # LIFO free list
        self.free_set = set(range(max_num_blocks))       # O(1) membership
        self._allocated = 0
        
    def allocate(self) -> int:
        """Allocate one physical block. Returns block index."""
        if not self.free_blocks:
            raise RuntimeError(f"Out of KV cache blocks ({self.max} max)")
        blk = self.free_blocks.pop()
        self.free_set.discard(blk)
        self._allocated += 1
        return blk
    
    def allocate_n(self, n: int) -> List[int]:
        """Allocate n physical blocks."""
        if len(self.free_blocks) < n:
            raise RuntimeError(f"Need {n} blocks, only {len(self.free_blocks)} free (max {self.max})")
        blocks = [self.allocate() for _ in range(n)]
        return blocks
    
    def free(self, block_idx: int):
        """Free a physical block."""
        if block_idx not in self.free_set:
            self.free_blocks.append(block_idx)
            self.free_set.add(block_idx)
            self._allocated -= 1
    
    def free_n(self, block_indices: List[int]):
        """Free multiple blocks."""
        for b in block_indices:
            self.free(b)
    
    @property
    def num_free(self) -> int:
        return len(self.free_blocks)
    
    @property
    def num_allocated(self) -> int:
        return self._allocated
    
    @property
    def usage_ratio(self) -> float:
        return self._allocated / self.max if self.max > 0 else 0.0


class PagedKVCache:
    """
    Paged KV cache.
    
    Physical memory layout:
      kv_cache[layer] = Tensor [max_num_blocks, block_size, num_kv_heads, head_dim]
      (K and V stored separately, or interleaved)
    
    Block table per sequence maps logical block index → physical block index.
    
    Usage:
      cache = PagedKVCache(config)
      blocks = cache.alloc_blocks(seq_id, num_tokens)
      cache.store_kv(layer_idx, seq_id, positions, k, v)
      k, v = cache.get_kv(layer_idx, seq_id)
      cache.free_sequence(seq_id)
    """
    
    def __init__(self, config: PagedKVConfig):
        self.cfg = config
        self.block_table = BlockTable(config.max_num_blocks)
        
        # Physical KV storage: one tensor per layer
        # Shape: [max_num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache = nn.ParameterList([
            nn.Parameter(
                torch.zeros(config.max_num_blocks, config.block_size,
                           config.num_kv_heads, config.head_dim,
                           dtype=config.dtype, device='cpu'),
                requires_grad=False
            ) for _ in range(config.num_layers)
        ])
        self.v_cache = nn.ParameterList([
            nn.Parameter(
                torch.zeros(config.max_num_blocks, config.block_size,
                           config.num_kv_heads, config.head_dim,
                           dtype=config.dtype, device='cpu'),
                requires_grad=False
            ) for _ in range(config.num_layers)
        ])
        
        # Per-sequence: list of allocated physical block indices
        self.seq_blocks: Dict[int, List[int]] = {}
        # Per-sequence: number of tokens stored
        self.seq_lengths: Dict[int, int] = {}
        # Next sequence ID
        self._next_seq_id = 0
        
    def new_sequence(self) -> int:
        """Create a new sequence, return sequence ID."""
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self.seq_blocks[seq_id] = []
        self.seq_lengths[seq_id] = 0
        return seq_id
    
    def reserve_blocks(self, seq_id: int, num_tokens: int):
        """Pre-allocate blocks for a sequence expecting num_tokens tokens."""
        needed = (num_tokens + self.cfg.block_size - 1) // self.cfg.block_size
        current = len(self.seq_blocks[seq_id])
        to_alloc = needed - current
        if to_alloc > 0:
            new_blocks = self.block_table.allocate_n(to_alloc)
            self.seq_blocks[seq_id].extend(new_blocks)
    
    def append_kv(self, seq_id: int, positions: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor):
        """
        Store KV for new tokens. k, v shapes: [num_new_tokens, num_kv_heads, head_dim]
        """
        num_new = k.shape[0]
        start_pos = self.seq_lengths[seq_id]
        end_pos = start_pos + num_new
        
        # Ensure we have enough blocks
        self.reserve_blocks(seq_id, end_pos)
        
        blocks = self.seq_blocks[seq_id]
        bs = self.cfg.block_size
        
        for offset in range(num_new):
            global_pos = start_pos + offset
            blk_idx = global_pos // bs
            blk_offset = global_pos % bs
            
            if blk_idx >= len(blocks):
                # Need more blocks
                new_blk = self.block_table.allocate()
                self.seq_blocks[seq_id].append(new_blk)
                blocks = self.seq_blocks[seq_id]
            
            phys_blk = blocks[blk_idx]
            
            for layer_idx in range(self.cfg.num_layers):
                self.k_cache[layer_idx].data[phys_blk, blk_offset] = k[layer_idx][offset]
                self.v_cache[layer_idx].data[phys_blk, blk_offset] = v[layer_idx][offset]
        
        self.seq_lengths[seq_id] = end_pos
    
    def store_kv(self, layer_idx: int, seq_id: int,
                 positions: torch.Tensor,
                 k: torch.Tensor, v: torch.Tensor):
        """
        Store KV for a specific layer.
        k, v: [num_tokens, num_kv_heads, head_dim]
        """
        num_tokens = k.shape[0]
        blocks = self.seq_blocks[seq_id]
        bs = self.cfg.block_size
        start = self.seq_lengths.get(seq_id, 0) if positions[0].item() != 0 else 0
        
        # Determine absolute positions
        if positions[0].item() == 0:
            # Starting fresh — clear old blocks
            self.seq_lengths[seq_id] = 0
        
        for i in range(num_tokens):
            pos = self.seq_lengths[seq_id] + i
            blk_idx = pos // bs
            blk_offset = pos % bs
            
            if blk_idx >= len(blocks):
                new_blk = self.block_table.allocate()
                self.seq_blocks[seq_id].append(new_blk)
                blocks = self.seq_blocks[seq_id]
            
            phys_blk = blocks[blk_idx]
            self.k_cache[layer_idx].data[phys_blk, blk_offset] = k[i]
            self.v_cache[layer_idx].data[phys_blk, blk_offset] = v[i]
        
        self.seq_lengths[seq_id] = max(self.seq_lengths.get(seq_id, 0),
                                        positions[-1].item() + 1)
    
    def get_kv(self, layer_idx: int, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full KV for a sequence (used during prefill).
        Returns (k, v) each [seq_len, num_kv_heads, head_dim]
        """
        seq_len = self.seq_lengths.get(seq_id, 0)
        if seq_len == 0:
            return (
                torch.empty(0, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=self.cfg.dtype),
                torch.empty(0, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=self.cfg.dtype),
            )
        
        blocks = self.seq_blocks[seq_id]
        bs = self.cfg.block_size
        
        k_out = torch.zeros(seq_len, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=self.cfg.dtype)
        v_out = torch.zeros(seq_len, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=self.cfg.dtype)
        
        for pos in range(seq_len):
            blk_idx = pos // bs
            blk_offset = pos % bs
            phys_blk = blocks[blk_idx]
            k_out[pos] = self.k_cache[layer_idx].data[phys_blk, blk_offset]
            v_out[pos] = self.v_cache[layer_idx].data[phys_blk, blk_offset]
        
        return k_out, v_out
    
    def get_kv_slice(self, layer_idx: int, seq_id: int,
                     start_pos: int, end_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV slice for a range of positions."""
        length = end_pos - start_pos
        blocks = self.seq_blocks[seq_id]
        bs = self.cfg.block_size
        
        k_out = torch.zeros(length, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=self.cfg.dtype)
        v_out = torch.zeros(length, self.cfg.num_kv_heads, self.cfg.head_dim, dtype=self.cfg.dtype)
        
        for i, pos in enumerate(range(start_pos, end_pos)):
            blk_idx = pos // bs
            blk_offset = pos % bs
            phys_blk = blocks[blk_idx]
            k_out[i] = self.k_cache[layer_idx].data[phys_blk, blk_offset]
            v_out[i] = self.v_cache[layer_idx].data[phys_blk, blk_offset]
        
        return k_out, v_out
    
    def free_sequence(self, seq_id: int):
        """Free all blocks allocated to a sequence."""
        if seq_id in self.seq_blocks:
            self.block_table.free_n(self.seq_blocks[seq_id])
            del self.seq_blocks[seq_id]
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]
    
    def free_old_blocks(self, seq_id: int, keep_last_n: int):
        """Free oldest blocks, keeping only the last N tokens."""
        if seq_id not in self.seq_blocks:
            return
        
        seq_len = self.seq_lengths.get(seq_id, 0)
        if seq_len <= keep_last_n:
            return
        
        bs = self.cfg.block_size
        cutoff = seq_len - keep_last_n
        first_keep_block = cutoff // bs
        
        blocks = self.seq_blocks[seq_id]
        to_free = blocks[:first_keep_block]
        if to_free:
            self.block_table.free_n(to_free)
            self.seq_blocks[seq_id] = blocks[first_keep_block:]
            self.seq_lengths[seq_id] = keep_last_n
    
    def to_device(self, device: torch.device):
        """Move all KV caches to a device."""
        for i in range(self.cfg.num_layers):
            self.k_cache[i] = nn.Parameter(self.k_cache[i].data.to(device), requires_grad=False)
            self.v_cache[i] = nn.Parameter(self.v_cache[i].data.to(device), requires_grad=False)
    
    def zero_block(self, phys_blk: int):
        """Zero out a physical block (for secure deallocation)."""
        for layer_idx in range(self.cfg.num_layers):
            self.k_cache[layer_idx].data[phys_blk].zero_()
            self.v_cache[layer_idx].data[phys_blk].zero_()
    
    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            'total_blocks': self.cfg.max_num_blocks,
            'free_blocks': self.block_table.num_free,
            'allocated_blocks': self.block_table.num_allocated,
            'usage_ratio': self.block_table.usage_ratio,
            'num_sequences': len(self.seq_blocks),
            'total_tokens': sum(self.seq_lengths.values()),
        }


# ═══════════════════════════════════════════════════════════════
# Paged Attention — Forward Implementation
# ═══════════════════════════════════════════════════════════════

def paged_attention_forward(
    query: torch.Tensor,           # [batch, num_heads, head_dim]
    kv_cache: PagedKVCache,
    layer_idx: int,
    seq_ids: List[int],
    positions: torch.Tensor,       # [batch]
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Paged attention: compute attention using paged KV cache.
    
    For each sequence in batch:
      1. Look up its block table
      2. Gather K, V from physical blocks
      3. Compute attention with query
    
    Args:
        query: [batch_size, num_heads, head_dim]
        kv_cache: PagedKVCache instance
        layer_idx: which transformer layer
        seq_ids: sequence IDs for each batch element
        positions: current positions [batch_size]
        sm_scale: 1/sqrt(head_dim), computed if None
    
    Returns:
        output: [batch_size, num_heads, head_dim]
    """
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    outputs = []
    
    for b in range(batch_size):
        seq_id = seq_ids[b]
        seq_len = kv_cache.seq_lengths.get(seq_id, 0)
        
        if seq_len == 0:
            # No KV cache yet — self-attention or first token
            outputs.append(torch.zeros(num_heads, head_dim, dtype=query.dtype))
            continue
        
        # Gather K, V for this sequence
        k, v = kv_cache.get_kv(layer_idx, seq_id)  # [seq_len, num_kv_heads, head_dim]
        
        # Handle GQA: expand KV heads if needed
        num_kv_heads = k.shape[1]
        if num_heads > num_kv_heads:
            # Repeat KV heads to match Q heads
            ratio = num_heads // num_kv_heads
            k = k.repeat_interleave(ratio, dim=1)  # [seq_len, num_heads, head_dim]
            v = v.repeat_interleave(ratio, dim=1)
        
        # Compute attention
        q = query[b]  # [num_heads, head_dim]
        # scores: [num_heads, seq_len]
        scores = torch.matmul(q, k.transpose(1, 2)) * sm_scale  # [num_heads, seq_len]
        
        # Causal mask (only needed for prefill)
        pos = positions[b].item()
        causal_mask = torch.arange(seq_len, device=scores.device) <= pos
        scores[:, ~causal_mask] = float('-inf')
        
        attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, seq_len]
        out = torch.matmul(attn_weights, v)  # [num_heads, head_dim]
        outputs.append(out)
    
    return torch.stack(outputs, dim=0)


# ═══════════════════════════════════════════════════════════════
# Prefix Sharing
# ═══════════════════════════════════════════════════════════════

class PrefixCache:
    """
    Caches KV blocks for common prefixes (system prompts, few-shot examples).
    
    When multiple requests share the same prefix, only compute KV once.
    """
    
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        # prefix_hash → list of physical block indices
        self.prefix_blocks: Dict[str, List[int]] = {}
        # prefix_hash → KV data (for lookup)
        self.prefix_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def store_prefix(self, prefix_hash: str, blocks: List[int],
                     k: torch.Tensor, v: torch.Tensor):
        """Store KV for a prefix."""
        self.prefix_blocks[prefix_hash] = blocks
        self.prefix_data[prefix_hash] = (k.clone(), v.clone())
    
    def get_prefix(self, prefix_hash: str) -> Optional[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        """Look up cached prefix. Returns (blocks, k, v) or None."""
        if prefix_hash in self.prefix_blocks:
            blocks = self.prefix_blocks[prefix_hash]
            k, v = self.prefix_data[prefix_hash]
            return blocks, k.clone(), v.clone()
        return None
    
    def evict_lru(self, max_prefixes: int = 100):
        """Evict least recently used prefixes."""
        if len(self.prefix_blocks) > max_prefixes:
            # Simple: evict oldest
            oldest = next(iter(self.prefix_blocks))
            del self.prefix_blocks[oldest]
            del self.prefix_data[oldest]


if __name__ == '__main__':
    # Quick test
    cfg = PagedKVConfig(num_layers=2, num_heads=4, num_kv_heads=2, head_dim=32,
                        block_size=8, max_num_blocks=64, dtype=torch.float32)
    cache = PagedKVCache(cfg)
    
    # Create a sequence and store KV
    seq_id = cache.new_sequence()
    k = torch.randn(3, 2, 2, 32)  # [layers, tokens, kv_heads, head_dim]
    v = torch.randn(3, 2, 2, 32)
    
    for layer in range(3):
        cache.store_kv(0, seq_id, torch.arange(2), k[layer], v[layer])
    
    print(f"Stats: {cache.stats()}")
    print(f"Seq {seq_id}: {cache.seq_lengths[seq_id]} tokens, {len(cache.seq_blocks[seq_id])} blocks")
    print("✅ PagedKVCache test passed")
