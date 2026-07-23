#!/usr/bin/env python3
"""
Phase 7 CLI — Distributed AI Platform.

Usage:
  # Single-node multi-GPU (torchrun)
  torchrun --nproc_per_node=8 -m agent.phase7.cli train --config small --data /path/to/data

  # Multi-node
  torchrun --nnodes=4 --nproc_per_node=8 --rdzv_endpoint=host:29400 \\
           -m agent.phase7.cli train --config xlarge --data /mnt/data

  # Launch with custom parallelism
  python -m agent.phase7.cli launch --gpus 8 --tp 2 --pp 2 --dp 2
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def cmd_train(args):
    """Launch distributed training."""
    from agent.phase7.distributed_trainer import DistributedTrainer, DistributedTrainingConfig
    from agent.phase7.parallelism import DistributedConfig
    from model import create_model

    # Parse world info from environment (set by torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Parse parallelism strategy
    tp = args.tensor_parallel or 1
    pp = args.pipeline_parallel or 1
    dp = world_size // (tp * pp)

    dist_cfg = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        dp_size=dp,
        tp_size=tp,
        pp_size=pp,
        backend="nccl",
    )

    # Create model
    model = create_model(args.config)

    # Training config
    train_cfg = DistributedTrainingConfig(
        model_config=args.config,
        batch_size_per_gpu=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        distributed=dist_cfg,
        use_fsdp=not args.no_fsdp,
        use_deepspeed=args.deepspeed,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
    )

    trainer = DistributedTrainer(model, train_cfg)

    try:
        trainer.train()
    finally:
        trainer.cleanup()


def cmd_benchmark(args):
    """Benchmark distributed training throughput."""
    import torch
    from agent.phase7.parallelism import DistributedConfig
    from agent.phase7.distributed_trainer import DistributedTrainer, DistributedTrainingConfig
    from model import create_model

    model = create_model(args.config)
    model.eval()

    # Create dummy data
    batch = {
        "input_ids": torch.randint(0, 32000, (args.batch_size, args.seq_len)),
        "labels": torch.randint(0, 32000, (args.batch_size, args.seq_len)),
    }

    # Warmup
    for _ in range(5):
        _ = model(batch["input_ids"])

    # Benchmark
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    n_steps = args.steps
    for _ in range(n_steps):
        _ = model(batch["input_ids"])
    torch.cuda.synchronize()
    dt = time.time() - t0

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    tok_per_sec = (args.batch_size * args.seq_len * n_steps) / dt
    tflops = 6 * total_params * 1e9 * (args.batch_size * args.seq_len) / dt / 1e12  # rough estimate

    print(f"Model: {total_params:.1f}B params")
    print(f"Batch: {args.batch_size} × seq {args.seq_len}")
    print(f"Throughput: {tok_per_sec:.0f} tok/s")
    print(f"Est. TFLOPS: {tflops:.1f}")
    print(f"GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"Time per step: {dt/n_steps*1000:.0f} ms")


def cmd_info(args):
    """Print distributed training info."""
    from agent.phase7.parallelism import DistributedConfig
    from agent.phase7.distributed_trainer import DistributedTrainingConfig

    print("╔══ Phase 7: Distributed AI Platform ══╗")
    print("║ Supported:")
    print("║   - FSDP (Fully Sharded Data Parallel)")
    print("║   - DeepSpeed ZeRO Stage 1/2/3")
    print("║   - Tensor Parallelism (Megatron-style)")
    print("║   - Pipeline Parallelism (1F1B schedule)")
    print("║   - Expert Parallelism (MoE)")
    print("║   - Mixed Precision: BF16, FP16, FP8")
    print("║   - NCCL All-Reduce/All-Gather/All-to-All")
    print("║   - Elastic Launch (torchrun)")
    print("║   - Distributed Checkpoint + Fault Recovery")
    print("║")
    print("║ Launch:")
    print("║   torchrun --nproc_per_node=8 -m agent.phase7.cli train --config xlarge")
    print("║")
    print("║ Custom parallelism (e.g. TP=2, PP=2 on 8 GPUs):")
    print("║   torchrun --nproc_per_node=8 -m agent.phase7.cli train --tp 2 --pp 2")
    print("╚═════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Distributed AI Platform")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Launch distributed training")
    train.add_argument("--config", default="small", help="Model config name")
    train.add_argument("--data", default="data/train.bin", help="Training data path")
    train.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    train.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    train.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train.add_argument("--max-steps", type=int, default=100000)
    train.add_argument("--tensor-parallel", type=int, default=0, help="Tensor parallel size")
    train.add_argument("--pipeline-parallel", type=int, default=0, help="Pipeline parallel size")
    train.add_argument("--output-dir", default="checkpoints")
    train.add_argument("--save-interval", type=int, default=5000)
    train.add_argument("--no-fsdp", action="store_true")
    train.add_argument("--deepspeed", action="store_true")

    bench = sub.add_parser("benchmark", help="Benchmark throughput")
    bench.add_argument("--config", default="small")
    bench.add_argument("--batch-size", type=int, default=2)
    bench.add_argument("--seq-len", type=int, default=8192)
    bench.add_argument("--steps", type=int, default=20)

    sub.add_parser("info", help="Show distributed training capabilities")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
