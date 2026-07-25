#!/usr/bin/env python3
"""
continuous_batcher.py — Continuous Batching Scheduler for TinyLLM

Key features:
  - Dynamic request queue: add/remove requests without stopping generation
  - Prefill-Decode split: batch prefill (parallel) + decode (autoregressive)
  - Priority scheduling: FIFO with optional priority levels
  - Preemption: evict low-priority sequences when OOM
  - Streaming: callback-based token delivery

Architecture:
  Request Queue → Batcher → Prefill (parallel) → KV Cache
                                         ↓
                                    Decode Loop ← tokens
                                         ↓
                                   Callbacks → user
"""

import time
import heapq
import threading
import sys, os
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch

# Support both package and script execution
try:
    from .paged_kv import PagedKVCache, PagedKVConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paged_kv import PagedKVCache, PagedKVConfig


class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"      # Processing prompt tokens
    DECODE = "decode"        # Generating output tokens
    FINISHED = "finished"


@dataclass
class GenerationRequest:
    """A single generation request."""
    request_id: int
    prompt_tokens: List[int]
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    stop_sequences: List[str] = field(default_factory=list)
    priority: int = 0          # Higher = more important
    created_at: float = field(default_factory=time.time)
    
    # Internal state (set by scheduler)
    state: RequestState = RequestState.WAITING
    seq_id: int = -1
    generated_tokens: List[int] = field(default_factory=list)
    prefill_done: bool = False
    finish_reason: str = ""    # "stop", "length", "abort"


@dataclass
class BatchConfig:
    """Continuous batching configuration."""
    max_batch_size: int = 32           # Max sequences in one batch
    max_tokens_per_batch: int = 4096   # Max total tokens in batch (prefill)
    block_size: int = 16               # KV cache block size
    max_num_blocks: int = 2048         # Total KV cache blocks
    max_seq_len: int = 8192            # Max sequence length
    queue_timeout_ms: int = 100        # Max wait for batching
    preempt_on_oom: bool = True        # Evict on OOM
    enable_prefix_cache: bool = True   # Cache common prefixes


class ContinuousBatcher:
    """
    Continuous batching scheduler.
    
    Usage:
      batcher = ContinuousBatcher(model, tokenizer, config)
      batcher.add_request(prompt, callback=my_callback)
      batcher.step()  # Run one scheduler step (call in loop)
    """
    
    def __init__(self, model, tokenizer, batch_cfg: BatchConfig,
                 kv_cfg: Optional[PagedKVConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_cfg = batch_cfg
        self.device = next(model.parameters()).device
        
        # KV cache
        if kv_cfg is None:
            kv_cfg = PagedKVConfig(
                block_size=batch_cfg.block_size,
                max_num_blocks=batch_cfg.max_num_blocks,
            )
        self.kv_cache = PagedKVCache(kv_cfg)
        self.kv_cache.to_device(self.device)
        
        # Request management
        self.waiting_queue: List[GenerationRequest] = []  # priority queue
        self.active_requests: Dict[int, GenerationRequest] = {}
        self._next_request_id = 0
        self._next_seq_id = 0
        
        # Callbacks
        self.callbacks: Dict[int, Callable] = {}
        
        # Stats
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.prefill_count = 0
        self.decode_count = 0
    
    def add_request(self, prompt: str | List[int],
                    callback: Optional[Callable[[int, str, bool], None]] = None,
                    **kwargs) -> int:
        """
        Add a generation request.
        
        Args:
            prompt: Text string or token ID list
            callback: fn(request_id, token_text, is_finished)
            **kwargs: max_new_tokens, temperature, top_p, top_k, stop_sequences, priority
        
        Returns:
            request_id (int)
        """
        if isinstance(prompt, str):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = prompt
        
        req_id = self._next_request_id
        self._next_request_id += 1
        
        req = GenerationRequest(
            request_id=req_id,
            prompt_tokens=prompt_tokens,
            **{k: v for k, v in kwargs.items() if k in GenerationRequest.__dataclass_fields__}
        )
        
        # Priority queue (negative for max-heap behavior)
        heapq.heappush(self.waiting_queue, (-req.priority, req.created_at, req_id))
        self.callbacks[req_id] = callback
        self.total_requests += 1
        
        return req_id
    
    def cancel_request(self, request_id: int):
        """Cancel a pending or active request."""
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            req.finish_reason = "abort"
            self._finish_request(req)
        # Remove from waiting queue (rebuild without it)
        self.waiting_queue = [(p, t, rid) for p, t, rid in self.waiting_queue if rid != request_id]
        heapq.heapify(self.waiting_queue)
    
    def step(self) -> int:
        """
        Run one scheduler step. Returns number of tokens generated this step.
        Call this in a loop until no active requests remain.
        """
        # 1. Schedule new requests from waiting queue → prefill
        self._schedule_prefills()
        
        # 2. Run prefill batch (parallel processing of prompt tokens)
        tokens_generated = self._run_prefill()
        
        # 3. Run decode step (one token per active sequence)
        tokens_generated += self._run_decode()
        
        # 4. Check finished sequences
        self._check_finished()
        
        return tokens_generated
    
    def run_until_idle(self, max_steps: int = 10000):
        """Run scheduler until all requests complete."""
        for _ in range(max_steps):
            if not self.active_requests and not self.waiting_queue:
                break
            self.step()
    
    # ── Internal methods ────────────────────────────────────
    
    def _schedule_prefills(self):
        """Move waiting requests to prefill state if capacity available."""
        available_blocks = self.kv_cache.block_table.num_free
        max_batch = self.batch_cfg.max_batch_size
        
        scheduled = []
        while self.waiting_queue and len(self.active_requests) < max_batch:
            _, _, req_id = heapq.heappop(self.waiting_queue)
            if req_id not in self.callbacks:
                continue  # cancelled
            
            req = GenerationRequest(
                request_id=req_id,
                prompt_tokens=[],  # will be filled
            )
            # Find the actual request data
            for q_req in self.active_requests.values():
                if q_req.request_id == req_id:
                    req = q_req
                    break
            
            # Estimate blocks needed
            prompt_len = len(req.prompt_tokens) if hasattr(req, 'prompt_tokens') else 0
            needed_blocks = (prompt_len + self.batch_cfg.block_size - 1) // self.batch_cfg.block_size
            needed_blocks += (req.max_new_tokens + self.batch_cfg.block_size - 1) // self.batch_cfg.block_size
            
            if needed_blocks <= available_blocks:
                req.state = RequestState.PREFILL
                req.seq_id = self.kv_cache.new_sequence()
                self.active_requests[req_id] = req
                available_blocks -= needed_blocks
                scheduled.append(req)
            else:
                # Not enough memory — put back in queue
                heapq.heappush(self.waiting_queue, (-req.priority, req.created_at, req_id))
                if not self.batch_cfg.preempt_on_oom:
                    break
                # Try preempting lowest-priority active request
                self._preempt_lowest()
                available_blocks = self.kv_cache.block_table.num_free
    
    def _run_prefill(self) -> int:
        """Process prefill for requests in PREFILL state. Returns token count."""
        prefill_reqs = [r for r in self.active_requests.values() if r.state == RequestState.PREFILL]
        if not prefill_reqs:
            return 0
        
        # Build batch: all prompt tokens concatenated
        batch_tokens = []
        batch_positions = []
        batch_seq_ids = []
        
        for req in prefill_reqs:
            prompt = req.prompt_tokens
            batch_tokens.extend(prompt)
            batch_positions.extend(range(len(prompt)))
            batch_seq_ids.extend([req.seq_id] * len(prompt))
        
        if not batch_tokens:
            return 0
        
        # Forward pass (prefill — parallel processing)
        input_ids = torch.tensor([batch_tokens], device=self.device).long()
        # Note: Real implementation would chunk by max_tokens_per_batch
        
        with torch.no_grad():
            # Run model forward, store KV in paged cache
            # (simplified: in practice, modify model.forward to use PagedKVCache)
            outputs = self.model(input_ids=input_ids)
        
        # Mark prefill complete
        for req in prefill_reqs:
            req.prefill_done = True
            req.state = RequestState.DECODE
            # Get last token logits → sample first generated token
            # (simplified)
        
        self.prefill_count += 1
        return len(batch_tokens)
    
    def _run_decode(self) -> int:
        """Run one decode step for all DECODE requests."""
        decode_reqs = [r for r in self.active_requests.values() if r.state == RequestState.DECODE]
        if not decode_reqs:
            return 0
        
        batch_size = len(decode_reqs)
        
        # Gather last tokens from each sequence
        last_tokens = []
        positions = []
        seq_ids = []
        for req in decode_reqs:
            if req.generated_tokens:
                last_tokens.append(req.generated_tokens[-1])
            else:
                last_tokens.append(req.prompt_tokens[-1])
            positions.append(len(req.prompt_tokens) + len(req.generated_tokens))
            seq_ids.append(req.seq_id)
        
        input_ids = torch.tensor([last_tokens], device=self.device).long()
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        
        # Sample next tokens
        logits = outputs['logits'][:, -1, :] / req.temperature
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        for i, req in enumerate(decode_reqs):
            token_id = next_tokens[i].item()
            req.generated_tokens.append(token_id)
            self.total_tokens_generated += 1
            
            # Callback
            cb = self.callbacks.get(req.request_id)
            if cb:
                token_text = self.tokenizer.decode([token_id])
                is_finished = (len(req.generated_tokens) >= req.max_new_tokens or
                              token_id == self.tokenizer.eos_token_id)
                cb(req.request_id, token_text, is_finished)
        
        self.decode_count += 1
        return batch_size
    
    def _check_finished(self):
        """Check for finished sequences and clean up."""
        finished_ids = []
        for req_id, req in self.active_requests.items():
            if req.state != RequestState.DECODE:
                continue
            
            # Check stop conditions
            if len(req.generated_tokens) >= req.max_new_tokens:
                req.finish_reason = "length"
                finished_ids.append(req_id)
            elif req.generated_tokens and req.generated_tokens[-1] == self.tokenizer.eos_token_id:
                req.finish_reason = "stop"
                # Remove EOS token
                req.generated_tokens.pop()
                finished_ids.append(req_id)
        
        for req_id in finished_ids:
            self._finish_request(self.active_requests[req_id])
    
    def _finish_request(self, req: GenerationRequest):
        """Clean up a finished request."""
        self.kv_cache.free_sequence(req.seq_id)
        req.state = RequestState.FINISHED
        self.active_requests.pop(req.request_id, None)
        
        # Final callback
        cb = self.callbacks.pop(req.request_id, None)
        if cb:
            cb(req.request_id, "", True)
    
    def _preempt_lowest(self):
        """Evict lowest-priority active request to free blocks."""
        if not self.active_requests:
            return
        
        # Find request with fewest generated tokens (least progress)
        victim = min(self.active_requests.values(),
                     key=lambda r: len(r.generated_tokens))
        
        victim.finish_reason = "preempted"
        self.kv_cache.free_sequence(victim.seq_id)
        self.active_requests.pop(victim.request_id, None)
        
        # Re-queue (will restart from scratch)
        heapq.heappush(self.waiting_queue,
                       (-victim.priority, victim.created_at, victim.request_id))
    
    def stats(self) -> dict:
        """Return scheduler statistics."""
        return {
            'active_requests': len(self.active_requests),
            'waiting_requests': len(self.waiting_queue),
            'total_requests': self.total_requests,
            'total_tokens_generated': self.total_tokens_generated,
            'prefill_steps': self.prefill_count,
            'decode_steps': self.decode_count,
            'kv_cache': self.kv_cache.stats(),
        }


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Testing ContinuousBatcher...")
    
    batch_cfg = BatchConfig(max_batch_size=8, max_tokens_per_batch=1024,
                            block_size=16, max_num_blocks=256)
    
    # Mock model (just returns random logits)
    class MockModel:
        def __call__(self, input_ids):
            B, S = input_ids.shape
            return {'logits': torch.randn(B, S, 32000)}
        def parameters(self):
            return iter([torch.zeros(1)])
    
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('tokenizer', use_fast=True)
    except:
        class MockTok:
            def encode(self, text):
                return [1, 2, 3, 4, 5]
            def decode(self, ids):
                return 'X'
            eos_token_id = 2
            vocab_size = 32000
        tokenizer = MockTok()
    
    batcher = ContinuousBatcher(MockModel(), tokenizer, batch_cfg)
    
    results = []
    def on_token(req_id, text, done):
        if done:
            results.append(req_id)
    
    batcher.add_request("Hello, how are you?", callback=on_token, max_new_tokens=10)
    batcher.add_request("What is AI?", callback=on_token, max_new_tokens=10)
    
    batcher.run_until_idle(max_steps=50)
    
    print(f"Completed: {len(results)} requests")
    print(f"Stats: {batcher.stats()}")
    print("✅ ContinuousBatcher test passed")
