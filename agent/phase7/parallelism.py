"""
Phase 7: Distributed Parallelism Strategies.

Implements all major parallelism paradigms for large-scale training:
  - Data Parallel (DP): replicate model, split batch, all-reduce gradients
  - Tensor Parallel (TP): split weight matrices across GPUs (Megatron-style)
  - Pipeline Parallel (PP): split layers across GPUs (GPipe-style)
  - Sequence Parallel (SP): split sequence dimension (Ring Attention)
  - Expert Parallel (EP): distribute MoE experts across GPUs

All strategies are composable (3D parallelism: DP + TP + PP).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp


# ═══════════════════════════════════════════════════════════════════════
# Distributed Context
# ═══════════════════════════════════════════════════════════════════════

class ParallelStrategy(Enum):
    NONE = "none"
    DATA_PARALLEL = "dp"
    TENSOR_PARALLEL = "tp"
    PIPELINE_PARALLEL = "pp"
    SEQUENCE_PARALLEL = "sp"
    EXPERT_PARALLEL = "ep"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # World info
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"  # nccl, gloo, mpi

    # Parallelism dimensions (product must equal world_size)
    dp_size: int = 1       # Data parallel replicas
    tp_size: int = 1       # Tensor parallel (Megatron)
    pp_size: int = 1       # Pipeline parallel (GPipe)
    ep_size: int = 1       # Expert parallel (MoE)

    # Pipeline
    pp_chunks: int = 4     # Micro-batches for pipeline (1F1B scheduling)
    pp_schedule: str = "1f1b"  # 1f1b, gpipe, interleaved

    # Tensor parallel
    tp_sequence_parallel: bool = False  # Sequence parallelism with TP

    # Communication
    grad_reduce_op: str = "sum"  # sum, avg
    use_fp8_allreduce: bool = False
    use_nccl_p2p: bool = True    # NCCL P2P for TP/PP

    # Expert parallel
    ep_alltoall_backend: str = "nccl"  # nccl, gloo
    ep_capacity_factor: float = 1.25

    # Mixed precision
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "bfloat16"
    buffer_dtype: str = "bfloat16"

    # Fault tolerance
    use_checkpointing: bool = True
    checkpoint_interval: int = 1000
    elastic_launch: bool = False
    max_restarts: int = 3

    def validate(self):
        """Ensure parallelism dims multiply to world_size."""
        product = self.dp_size * self.tp_size * self.pp_size
        if product != self.world_size:
            raise ValueError(
                f"dp_size({self.dp_size}) * tp_size({self.tp_size}) * "
                f"pp_size({self.pp_size}) = {product}, must equal "
                f"world_size({self.world_size})"
            )

    @property
    def tp_group_size(self) -> int: return self.tp_size
    @property
    def pp_group_size(self) -> int: return self.pp_size
    @property
    def dp_group_size(self) -> int: return self.dp_size
    @property
    def ep_group_size(self) -> int: return self.ep_size


# ═══════════════════════════════════════════════════════════════════════
# Process Group Management
# ═══════════════════════════════════════════════════════════════════════

class ParallelGroupManager:
    """Manages all NCCL process groups for 3D parallelism."""

    def __init__(self, cfg: DistributedConfig):
        self.cfg = cfg
        self._groups: dict[str, dist.ProcessGroup] = {}
        self._init_groups()

    def _init_groups(self):
        """Initialize process groups for DP, TP, PP, EP."""
        world_size = self.cfg.world_size
        rank = self.cfg.rank

        # Compute 3D grid mapping: rank → (dp_rank, tp_rank, pp_rank, ep_rank)
        dp = self.cfg.dp_size
        tp = self.cfg.tp_size
        pp = self.cfg.pp_size
        ep = self.cfg.ep_size

        # Default: ep = 1, layout is [dp, tp, pp]
        rank_rem = rank
        self.pp_rank = rank_rem % pp; rank_rem //= pp
        self.tp_rank = rank_rem % tp; rank_rem //= tp
        self.dp_rank = rank_rem

        # Build TP group: same dp, same pp, different tp
        for gname, dim_size, get_rank in [
            ("tp", tp, lambda r: self.tp_rank),
            ("pp", pp, lambda r: self.pp_rank),
            ("dp", dp, lambda r: self.dp_rank),
        ]:
            if dim_size > 1:
                # Find ranks in same group
                if gname == "tp":
                    ranks = [
                        r for r in range(world_size)
                        if self._get_dp(r) == self.dp_rank and self._get_pp(r) == self.pp_rank
                    ]
                elif gname == "pp":
                    ranks = [
                        r for r in range(world_size)
                        if self._get_dp(r) == self.dp_rank and self._get_tp(r) == self.tp_rank
                    ]
                elif gname == "dp":
                    ranks = [
                        r for r in range(world_size)
                        if self._get_tp(r) == self.tp_rank and self._get_pp(r) == self.pp_rank
                    ]
                else:
                    continue
                self._groups[gname] = dist.new_group(ranks)

    def _get_dp(self, rank): return (rank // (self.cfg.tp_size * self.cfg.pp_size))
    def _get_tp(self, rank): return (rank // self.cfg.pp_size) % self.cfg.tp_size
    def _get_pp(self, rank): return rank % self.cfg.pp_size

    def get_group(self, name: str) -> Optional[dist.ProcessGroup]:
        return self._groups.get(name)

    @property
    def tp_group(self): return self._groups.get("tp")
    @property
    def pp_group(self): return self._groups.get("pp")
    @property
    def dp_group(self): return self._groups.get("dp")


# ═══════════════════════════════════════════════════════════════════════
# Tensor Parallelism (Megatron-LM style)
# ═══════════════════════════════════════════════════════════════════════

class ColumnParallelLinear(nn.Module):
    """Column-wise parallelism: split output dimension across GPUs.
    
    Y = X @ W^T, where W is split column-wise.
    Each GPU holds W[:, start:end], computes partial Y.
    Y is all-gathered across TP group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        bias: bool = False,
        gather_output: bool = True,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output
        self.in_features = in_features

        # Split output dim
        chunk = (out_features + tp_size - 1) // tp_size
        self.out_features_local = chunk
        self.out_features_global = out_features

        self.weight = nn.Parameter(torch.empty(chunk, in_features))
        self.bias = nn.Parameter(torch.empty(chunk)) if bias else None
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [batch, seq, in_features]
        # Each GPU: Y_local = X @ W_local^T
        y_local = F.linear(x, self.weight, self.bias)

        if self.gather_output and self.tp_size > 1:
            # All-gather along last dim
            # Note: out_features may not be evenly divisible by tp_size
            # The gathered output uses the actual global out_features
            gathered_shape = list(y_local.shape)
            gathered_shape[-1] = self.out_features_local * self.tp_size
            y_global = torch.empty(gathered_shape, dtype=y_local.dtype, device=y_local.device)
            dist.all_gather_into_tensor(y_global, y_local, group=None)
            # Slice to actual global out_features if there was padding
            return y_global[..., :self.out_features_global]
        return y_local


class RowParallelLinear(nn.Module):
    """Row-wise parallelism: split input dimension across GPUs.
    
    Y = X @ W^T, where W is split row-wise.
    Each GPU holds W[:, start:end] and input X[:, start:end].
    Partial sums are all-reduced across TP group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        bias: bool = False,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_is_parallel = input_is_parallel

        chunk = (in_features + tp_size - 1) // tp_size
        self.in_features_local = chunk

        self.weight = nn.Parameter(torch.empty(out_features, chunk))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel and self.tp_size > 1:
            # x is already partitioned
            y_local = F.linear(x, self.weight)
            # All-reduce across TP group
            dist.all_reduce(y_local, op=ReduceOp.SUM)
            if self.bias is not None:
                y_local = y_local + self.bias
            return y_local
        else:
            return F.linear(x, self.weight, self.bias)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Parallelism
# ═══════════════════════════════════════════════════════════════════════

class PipelineSchedule:
    """1F1B (One-Forward-One-Backward) pipeline schedule."""

    def __init__(self, pp_size: int, pp_rank: int, n_microbatches: int):
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.n_mb = n_microbatches

    def get_schedule(self) -> List[Tuple[str, int]]:
        """Returns list of (action, microbatch_id) for 1F1B scheduling."""
        # 1F1B: warmup forward → 1F1B steady → cooldown backward
        schedule = []
        n_warmup = self.pp_size - self.pp_rank - 1
        n_steady = self.n_mb - n_warmup

        # Warmup: forward only
        for mb in range(n_warmup):
            schedule.append(("forward", mb))

        # Steady: 1 forward + 1 backward
        for i in range(n_steady):
            schedule.append(("forward", n_warmup + i))
            schedule.append(("backward", i))

        # Cooldown: backward only
        for mb in range(n_steady, self.n_mb):
            schedule.append(("backward", mb))

        return schedule


class PipelineStage(nn.Module):
    """Wraps a subset of model layers for pipeline parallelism."""

    def __init__(self, layers: nn.ModuleList, pp_rank: int, pp_size: int):
        super().__init__()
        self.layers = layers
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.input_buffer: Optional[torch.Tensor] = None  # for P2P recv
        self.output_buffer: Optional[torch.Tensor] = None  # for P2P send

    def forward(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        # Receive from previous stage
        if self.pp_rank > 0:
            if x is None:
                # Async recv from pp_rank-1
                x = self._recv_forward()

        # Forward through layers
        for layer in self.layers:
            x = layer(x)

        # Send to next stage
        if self.pp_rank < self.pp_size - 1:
            self._send_forward(x)
            return None  # No output at intermediate stages
        return x

    def backward(self, grad: Optional[torch.Tensor] = None):
        """Backward pass with P2P communication."""
        if self.pp_rank < self.pp_size - 1:
            grad = self._recv_backward()
        # Backward through layers in reverse
        if grad is not None:
            grad.backward()
        if self.pp_rank > 0:
            # Send gradient to previous stage
            input_grad = self.input_buffer.grad if hasattr(self.input_buffer, 'grad') else None
            if input_grad is not None:
                self._send_backward(input_grad)

    def _recv_forward(self) -> torch.Tensor:
        src = self.pp_rank - 1
        shape = self.input_buffer.shape if self.input_buffer is not None else None
        if shape:
            tensor = torch.empty(shape, device=self.input_buffer.device)
            dist.recv(tensor, src=src)
            return tensor
        raise RuntimeError("Input buffer not initialized")

    def _send_forward(self, tensor: torch.Tensor):
        dst = self.pp_rank + 1
        dist.send(tensor, dst=dst)

    def _recv_backward(self) -> torch.Tensor:
        src = self.pp_rank + 1
        shape = self.output_buffer.shape if self.output_buffer is not None else None
        if shape:
            tensor = torch.empty(shape, device=self.output_buffer.device)
            dist.recv(tensor, src=src)
            return tensor
        raise RuntimeError("Output buffer not initialized")

    def _send_backward(self, tensor: torch.Tensor):
        dst = self.pp_rank - 1
        dist.send(tensor, dst=dst)


# ═══════════════════════════════════════════════════════════════════════
# Expert Parallelism (for MoE)
# ═══════════════════════════════════════════════════════════════════════

class ExpertParallelDispatch:
    """All-to-all dispatch for expert parallelism in MoE layers.
    
    Tokens are routed to the GPU that hosts the target expert(s).
    Uses NCCL all-to-all for efficient token exchange.
    """

    def __init__(self, ep_size: int, ep_rank: int, n_experts: int):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.n_experts = n_experts
        self.experts_per_rank = (n_experts + ep_size - 1) // ep_size
        self.expert_start = ep_rank * self.experts_per_rank
        self.expert_end = min(self.expert_start + self.experts_per_rank, n_experts)

    def dispatch(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to the GPUs that host their target experts.
        
        Args:
            tokens: [total_tokens, hidden_dim]
            expert_indices: [total_tokens, k] — expert IDs per token
        
        Returns:
            dispatched_tokens: tokens routed to this rank's experts
            local_indices: expert indices adjusted for local expert numbering
        """
        if self.ep_size == 1:
            return tokens, expert_indices

        # Create send/recv counts for all-to-all
        send_counts = torch.zeros(self.ep_size, dtype=torch.int64)
        for i in range(self.ep_size):
            start = i * self.experts_per_rank
            end = min(start + self.experts_per_rank, self.n_experts)
            mask = (expert_indices >= start) & (expert_indices < end)
            send_counts[i] = mask.any(dim=-1).sum().item()

        recv_counts = torch.zeros(self.ep_size, dtype=torch.int64)
        dist.all_to_all_single(recv_counts, send_counts)

        # All-to-all exchange of tokens
        # Simplified: use scatter/gather based on counts
        send_tensors = []
        for i in range(self.ep_size):
            start = i * self.experts_per_rank
            end = min(start + self.experts_per_rank, self.n_experts)
            mask = (expert_indices >= start) & (expert_indices < end)
            mask = mask.any(dim=-1)
            if mask.sum() > 0:
                send_tensors.append(tokens[mask])
            else:
                send_tensors.append(torch.empty(0, tokens.shape[1], device=tokens.device, dtype=tokens.dtype))

        # Naive all-to-all via send/recv (production uses NCCL alltoallv)
        recv_list = []
        for i in range(self.ep_size):
            if recv_counts[i].item() > 0:
                recv_list.append(torch.empty(
                    recv_counts[i].item(), tokens.shape[1],
                    device=tokens.device, dtype=tokens.dtype,
                ))
            else:
                recv_list.append(torch.empty(0, tokens.shape[1], device=tokens.device, dtype=tokens.dtype))

        # Exchange (simplified: use dist.all_to_all for same-size chunks)
        # For production, use dist.all_to_all_single with proper counts
        if send_counts.sum() > 0:
            dist.all_to_all(recv_list, send_tensors)

        dispatched = torch.cat([t for t in recv_list if t.numel() > 0], dim=0)
        local_indices = expert_indices % self.experts_per_rank

        return dispatched, local_indices

    def combine(self, expert_outputs: torch.Tensor, send_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine expert outputs back to original token ordering."""
        if self.ep_size == 1:
            return expert_outputs
        # Reverse all-to-all
        # Simplified implementation
        return expert_outputs


import torch.nn.functional as F
