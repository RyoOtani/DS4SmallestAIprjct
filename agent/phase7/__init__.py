"""
Phase 7: Distributed AI Platform.

Modules:
- parallelism           — Data/Tensor/Pipeline/Sequence/Expert parallel strategies
- distributed_trainer   — FSDP, DeepSpeed ZeRO-3, NCCL, elastic launch
- mixed_precision       — BF16, FP16, FP8, dynamic loss scaling
- nccl_ops              — NCCL collectives, distributed checkpoint, elastic resume
- cli                   — CLI for distributed training
"""

from .parallelism import (
    DistributedConfig, ParallelGroupManager,
    ColumnParallelLinear, RowParallelLinear,
    PipelineStage, PipelineSchedule,
    ExpertParallelDispatch, ParallelStrategy,
)
from .distributed_trainer import DistributedTrainer, DistributedTrainingConfig
from .mixed_precision import MixedPrecisionManager, MixedPrecisionConfig, FP8Handler, DynamicLossScaler
from .nccl_ops import (
    all_reduce, all_gather, reduce_scatter, all_to_all,
    broadcast, barrier, send, recv,
    DistributedCheckpointManager, ElasticCheckpoint,
    sync_gradients, sync_batch_norm, broadcast_model_params,
    all_reduce_dict,
)

__all__ = [
    "DistributedConfig", "ParallelGroupManager",
    "ColumnParallelLinear", "RowParallelLinear",
    "PipelineStage", "PipelineSchedule",
    "ExpertParallelDispatch", "ParallelStrategy",
    "DistributedTrainer", "DistributedTrainingConfig",
    "MixedPrecisionManager", "MixedPrecisionConfig", "FP8Handler", "DynamicLossScaler",
    "all_reduce", "all_gather", "reduce_scatter", "all_to_all",
    "broadcast", "barrier", "send", "recv",
    "DistributedCheckpointManager", "ElasticCheckpoint",
    "sync_gradients", "sync_batch_norm", "broadcast_model_params",
    "all_reduce_dict",
]
