"""
Phase 7: NCCL Collective Operations & Distributed Checkpoint.

Wraps NCCL primitives for:
  - All-Reduce (gradient sync)
  - All-Gather (tensor parallel output)
  - Reduce-Scatter (FSDP gradient sharding)
  - All-to-All (expert parallel token exchange)
  - Broadcast (model init sync)
  - P2P Send/Recv (pipeline parallel)
  - Barrier (synchronization)

Also provides distributed checkpoint save/restore with:
  - torch.distributed.checkpoint (DCP) — PyTorch native
  - Sharded checkpoint per-rank
  - Full checkpoint gather
  - Fault-tolerant resume with elastic launch
"""

from __future__ import annotations
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from typing import Optional, List
import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# NCCL Collective Wrappers
# ═══════════════════════════════════════════════════════════════════════

def all_reduce(tensor: torch.Tensor, op: str = "sum", group=None, async_op: bool = False):
    """All-reduce across a process group."""
    op_map = {"sum": ReduceOp.SUM, "avg": ReduceOp.AVG, "prod": ReduceOp.PRODUCT,
              "min": ReduceOp.MIN, "max": ReduceOp.MAX}
    return dist.all_reduce(tensor, op=op_map.get(op, ReduceOp.SUM), group=group, async_op=async_op)


def all_gather(tensor: torch.Tensor, group=None) -> List[torch.Tensor]:
    """All-gather: each rank contributes tensor, all receive concatenated list."""
    world_size = dist.get_world_size(group)
    output = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor, group=group)
    return output


def all_gather_into_tensor(output: torch.Tensor, tensor: torch.Tensor, group=None):
    """All-gather into a single pre-allocated tensor."""
    dist.all_gather_into_tensor(output, tensor, group=group)


def reduce_scatter(output: torch.Tensor, input_list: List[torch.Tensor], group=None):
    """Reduce-scatter: each rank gets a chunk of the reduced result."""
    dist.reduce_scatter(output, input_list, group=group)


def all_to_all(output: List[torch.Tensor], input_list: List[torch.Tensor], group=None):
    """All-to-all: scatter different data to each rank."""
    dist.all_to_all(output, input_list, group=group)


def broadcast(tensor: torch.Tensor, src: int = 0, group=None):
    """Broadcast tensor from src rank to all others."""
    dist.broadcast(tensor, src=src, group=group)


def barrier(group=None):
    """Synchronize all ranks."""
    dist.barrier(group=group)


def send(tensor: torch.Tensor, dst: int, group=None):
    """P2P send."""
    dist.send(tensor, dst=dst, group=group)


def recv(tensor: torch.Tensor, src: int, group=None):
    """P2P receive."""
    dist.recv(tensor, src=src, group=group)


def p2p_send_recv(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    send_dst: int,
    recv_src: int,
    group=None,
):
    """P2P send to one rank and receive from another simultaneously."""
    send_op = dist.P2POp(dist.send, send_tensor, send_dst, group)
    recv_op = dist.P2POp(dist.recv, recv_tensor, recv_src, group)
    dist.batch_isend_irecv([send_op, recv_op])


def get_rank(group=None) -> int:
    return dist.get_rank(group)


def get_world_size(group=None) -> int:
    return dist.get_world_size(group)


# ═══════════════════════════════════════════════════════════════════════
# Distributed Checkpointing
# ═══════════════════════════════════════════════════════════════════════

class DistributedCheckpointManager:
    """Manages distributed checkpoint save/load with fault tolerance."""

    def __init__(self, output_dir: str, keep_last_n: int = 5):
        self.output_dir = Path(output_dir)
        self.keep_last_n = keep_last_n
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        global_step: int,
        best_val_loss: float,
        tag: str = "",
    ):
        """Save distributed checkpoint: each rank saves its own shard."""
        rank = get_rank()
        tag = tag or str(global_step)
        ckpt_dir = self.output_dir / f"step_{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Each rank saves its shard
        shard_file = ckpt_dir / f"rank_{rank:04d}.pt"

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "rank": rank,
        }

        torch.save(state, shard_file)

        # Rank 0 saves metadata
        if rank == 0:
            meta = {
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "num_ranks": get_world_size(),
                "tag": tag,
            }
            torch.save(meta, ckpt_dir / "metadata.pt")

        barrier()
        self._prune_old_checkpoints()

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        tag: str = "latest",
    ) -> int:
        """Load distributed checkpoint."""
        ckpt_dir = self.output_dir
        if tag == "latest":
            # Find latest checkpoint
            dirs = sorted(ckpt_dir.glob("step_*"))
            if not dirs:
                if get_rank() == 0:
                    print("No checkpoint found. Starting from scratch.")
                return 0
            ckpt_dir = dirs[-1]

        # Load metadata
        meta_path = ckpt_dir / "metadata.pt"
        if not meta_path.exists():
            return 0

        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        global_step = meta["global_step"]

        # Load shard
        rank = get_rank()
        shard_file = ckpt_dir / f"rank_{rank:04d}.pt"
        if shard_file.exists():
            state = torch.load(shard_file, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=False)
            optimizer.load_state_dict(state["optimizer"])
            if scheduler and state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])

        barrier()
        return global_step

    def _prune_old_checkpoints(self):
        """Remove old checkpoints beyond keep_last_n."""
        if get_rank() != 0:
            return
        dirs = sorted(self.output_dir.glob("step_*"))
        if len(dirs) > self.keep_last_n:
            for old_dir in dirs[:-self.keep_last_n]:
                import shutil
                shutil.rmtree(old_dir)


class ElasticCheckpoint:
    """Checkpointing with elastic launch support (changing world size)."""

    def __init__(self, manager: DistributedCheckpointManager):
        self.manager = manager
        self.restart_count = 0

    def save_elastic(
        self, model, optimizer, scheduler, global_step: int, best_val_loss: float,
    ):
        """Save checkpoint compatible with elastic scaling."""
        # Save full unsharded state dict (can be loaded with different world size)
        if hasattr(model, '_fsdp_wrapped_module'):
            # Gather full state dict from FSDP
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with model.state_dict_type(StateDictType.FULL_STATE_DICT, cfg):
                full_state = model.state_dict()
            if get_rank() == 0:
                torch.save(full_state, self.manager.output_dir / f"elastic_step_{global_step}.pt")
        else:
            # Standard save
            self.manager.save(model, optimizer, scheduler, global_step, best_val_loss, f"elastic_{global_step}")

    def load_elastic(self, model, optimizer, scheduler) -> int:
        """Resume with potentially different world size."""
        elastic_files = sorted(self.manager.output_dir.glob("elastic_step_*.pt"))
        if not elastic_files:
            return self.manager.load(model, optimizer, scheduler)

        ckpt = torch.load(elastic_files[-1], map_location="cpu")
        model.load_state_dict(ckpt if not isinstance(ckpt, dict) or "global_step" not in ckpt else ckpt["model"], strict=False)
        return ckpt.get("global_step", 0) if isinstance(ckpt, dict) else 0


# ═══════════════════════════════════════════════════════════════════════
# Gradient Synchronization Utilities
# ═══════════════════════════════════════════════════════════════════════

def sync_gradients(model: torch.nn.Module, group=None):
    """Synchronize gradients across a process group (for manual DP)."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=ReduceOp.AVG, group=group)


def sync_batch_norm(model: torch.nn.Module, group=None):
    """Synchronize BatchNorm statistics across GPUs."""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if module.running_mean is not None:
                dist.all_reduce(module.running_mean, op=ReduceOp.AVG, group=group)
            if module.running_var is not None:
                dist.all_reduce(module.running_var, op=ReduceOp.AVG, group=group)
            if module.weight is not None:
                dist.all_reduce(module.weight, op=ReduceOp.AVG, group=group)
            if module.bias is not None:
                dist.all_reduce(module.bias, op=ReduceOp.AVG, group=group)


def broadcast_model_params(model: torch.nn.Module, src: int = 0, group=None):
    """Broadcast model parameters from src rank (for initialization)."""
    for param in model.parameters():
        dist.broadcast(param.data, src=src, group=group)


def all_reduce_dict(metrics: dict, group=None) -> dict:
    """All-reduce a dictionary of scalar metrics."""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            t = value.clone().detach()
            dist.all_reduce(t, op=ReduceOp.AVG, group=group)
            result[key] = t.item()
        else:
            result[key] = value
    return result
