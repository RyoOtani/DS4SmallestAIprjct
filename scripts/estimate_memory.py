#!/usr/bin/env python3
"""
estimate_memory.py — Memory & cost estimator for Hierarchical MoE training.

Usage:
    python scripts/estimate_memory.py                    # default (FP16, ZeRO-3)
    python scripts/estimate_memory.py --dtype fp32       # FP32 full precision
    python scripts/estimate_memory.py --zero-stage 2     # ZeRO Stage 2
    python scripts/estimate_memory.py --batch-size 4     # larger batch
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════
# Model config (matches HierarchicalMoEConfig defaults)
# ═══════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    hidden_dim: int = 2048
    n_layers: int = 26
    n_heads: int = 16
    head_dim: int = 128
    vocab_size: int = 72000
    n_moe_layers: int = 16
    n_domain_groups: int = 4
    n_experts_per_group: int = 6
    n_shared_experts: int = 2
    n_active_domains: int = 2
    n_active_experts: int = 2
    expert_ffn_dim: int = 5632
    max_seq_len: int = 4096

    @property
    def total_experts(self) -> int:
        return self.n_moe_layers * (
            self.n_domain_groups * self.n_experts_per_group + self.n_shared_experts
        )

    @property
    def active_experts_per_layer(self) -> int:
        return self.n_active_domains * self.n_active_experts + self.n_shared_experts


# ═══════════════════════════════════════════════════════
# GPU pricing (USD/hr, spot ≈ 40% of on-demand)
# ═══════════════════════════════════════════════════════

GPU_PRICES = {
    "A100-40GB":  {"on_demand": 3.06, "spot": 1.22},
    "A100-80GB":  {"on_demand": 4.10, "spot": 1.64},
    "H100-80GB":  {"on_demand": 5.98, "spot": 2.39},
    "H200-141GB": {"on_demand": 8.00, "spot": 3.20},
}


# ═══════════════════════════════════════════════════════
# Memory estimation
# ═══════════════════════════════════════════════════════

def estimate_params(spec: ModelSpec) -> dict:
    """Break down parameter counts."""
    D, F = spec.hidden_dim, spec.expert_ffn_dim
    n_dense = spec.n_layers - spec.n_moe_layers

    # Embedding
    emb = spec.vocab_size * D

    # Attention per layer: Q, K, V, O projections + optional biases
    attn_per_layer = 4 * D * D  # QKV + O projections
    # Note: MultiheadAttention has in_proj_weight [3*embed, embed] and out_proj [embed, embed]
    # Actually nn.MHA: in_proj_weight is [3*D, D], out_proj is [D, D] = 4*D^2
    attn_total = attn_per_layer * spec.n_layers

    # Dense FFN per layer: 3 linear layers (gate-like)
    dense_ffn_per_layer = 3 * D * F
    dense_ffn_total = dense_ffn_per_layer * n_dense

    # MoE experts: 3 matrices (gate, up, down) per expert
    expert_params_each = 3 * D * F
    total_expert = expert_params_each * spec.total_experts

    total = emb + attn_total + dense_ffn_total + total_expert
    active_expert = expert_params_each * spec.n_moe_layers * spec.active_experts_per_layer
    active_total = emb + attn_total + dense_ffn_total + active_expert

    return {
        "total": total,
        "active": active_total,
        "embedding": emb,
        "attention": attn_total,
        "dense_ffn": dense_ffn_total,
        "experts_total": total_expert,
        "experts_active": active_expert,
    }


def estimate_memory_gb(spec: ModelSpec, dtype: str = "fp16",
                       batch_size: int = 1, seq_len: int = 4096,
                       zero_stage: int = 3) -> dict:
    """Estimate GPU memory requirements for training.

    Memory breakdown (pre-ZeRO, per GPU):
      Model weights:  params × bytes_per_param
      Gradients:      params × bytes_per_param
      Optimizer:      params × 8 bytes (AdamW fp32 momentum + variance)
      Activations:    depends on batch, seq, hidden, layers

    ZeRO partitioning:
      Stage 1: optimizer state sharded → 1/N_gpus
      Stage 2: + gradients sharded     → 1/N_gpus
      Stage 3: + parameters sharded    → 1/N_gpus
    """
    params = estimate_params(spec)
    total_params = params["total"]
    active_params = params["active"]

    bytes_per_param = 2 if dtype == "fp16" else 4
    param_gb = total_params * bytes_per_param / 1e9

    # Pre-ZeRO memory
    weights_gb = total_params * bytes_per_param / 1e9
    gradients_gb = total_params * bytes_per_param / 1e9
    optimizer_gb = total_params * 8 / 1e9  # AdamW: fp32 momentum (4) + variance (4)

    # Activation memory estimate (rough)
    # Per transformer layer: ~ B × S × D × (34-50) bytes depending on attn impl
    D = spec.hidden_dim
    act_per_layer_mb = batch_size * seq_len * D * 34 / 1e6  # 34 bytes per elem (empirical)
    activations_gb = act_per_layer_mb * spec.n_layers / 1000

    base_total = weights_gb + gradients_gb + optimizer_gb + activations_gb

    # ZeRO reduction factors
    if zero_stage == 1:
        shard_factor_opt = 8  # optimizer states shared, but we assume 8 GPUs
        shard_factor_grad = 1
        shard_factor_param = 1
    elif zero_stage == 2:
        shard_factor_opt = 8
        shard_factor_grad = 8
        shard_factor_param = 1
    else:  # stage 3
        shard_factor_opt = 8
        shard_factor_grad = 8
        shard_factor_param = 8

    per_gpu_gb = (
        weights_gb / shard_factor_param
        + gradients_gb / shard_factor_grad
        + optimizer_gb / shard_factor_opt
        + activations_gb
    )

    # With CPU offload
    per_gpu_offload_gb = (
        weights_gb / shard_factor_param
        + gradients_gb / shard_factor_grad
        + activations_gb  # optimizer offloaded to CPU
        + 0.5  # CPU offload overhead
    )

    return {
        "total_params": total_params,
        "active_params": active_params,
        "dtype": dtype,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "zero_stage": zero_stage,
        "weights_gb": weights_gb,
        "gradients_gb": gradients_gb,
        "optimizer_gb": optimizer_gb,
        "activations_gb": activations_gb,
        "base_total_gb": base_total,
        "per_gpu_gb": per_gpu_gb,
        "per_gpu_offload_gb": per_gpu_offload_gb,
    }


def recommend_gpus(mem: dict) -> list[dict]:
    """Recommend GPU types and count based on memory estimate."""
    per_gpu_req = mem["per_gpu_offload_gb"] if mem["zero_stage"] == 3 else mem["per_gpu_gb"]
    recommendations = []

    for gpu_name, prices in GPU_PRICES.items():
        gpu_mem = int(gpu_name.split("-")[1].replace("GB", ""))
        n_gpus = math.ceil(per_gpu_req / (gpu_mem * 0.85))  # 85% usable
        n_gpus = max(n_gpus, 1)

        # For 8-GPU assumption in ZeRO, round up
        if mem["zero_stage"] >= 2:
            n_gpus = max(n_gpus, 1)

        hourly_on_demand = n_gpus * prices["on_demand"]
        hourly_spot = n_gpus * prices["spot"]

        # Training time estimate: ~200K steps × batch/step
        # Assume ~0.5 sec per step → ~28 hours for 200K steps
        est_hours = 28
        total_on_demand = hourly_on_demand * est_hours
        total_spot = hourly_spot * est_hours

        recommendations.append({
            "gpu": gpu_name,
            "gpu_mem_gb": gpu_mem,
            "n_gpus_min": n_gpus,
            "per_gpu_mem_gb": per_gpu_req,
            "mem_utilization": f"{per_gpu_req / (gpu_mem * 0.85) * 100:.0f}%",
            "hourly_on_demand": hourly_on_demand,
            "hourly_spot": hourly_spot,
            "est_total_on_demand": total_on_demand,
            "est_total_spot": total_spot,
        })

    return recommendations


def print_report(mem: dict, recommendations: list[dict]):
    """Pretty-print the estimation report."""
    print("=" * 70)
    print("🧠 Hierarchical MoE — Memory & Cost Estimator")
    print("=" * 70)

    print(f"\n📐 Model Spec")
    print(f"   Total params:    {mem['total_params']/1e9:.2f}B")
    print(f"   Active params:   {mem['active_params']/1e9:.2f}B")
    print(f"   Dtype:           {mem['dtype'].upper()}")
    print(f"   Batch/Seq:       {mem['batch_size']} × {mem['seq_len']}")
    print(f"   ZeRO Stage:      {mem['zero_stage']}")

    print(f"\n💾 Memory Breakdown (pre-ZeRO, single GPU)")
    print(f"   Weights:         {mem['weights_gb']:.1f} GB")
    print(f"   Gradients:       {mem['gradients_gb']:.1f} GB")
    print(f"   Optimizer:       {mem['optimizer_gb']:.1f} GB")
    print(f"   Activations:     {mem['activations_gb']:.1f} GB")
    print(f"   ────────────────────────")
    print(f"   BASE TOTAL:      {mem['base_total_gb']:.1f} GB  ⚠️  requires {math.ceil(mem['base_total_gb']/80)}× H100 without ZeRO")

    print(f"\n📦 Per-GPU Memory (ZeRO-{mem['zero_stage']}, 8-GPU assumption)")
    print(f"   No offload:      {mem['per_gpu_gb']:.1f} GB")
    if mem['zero_stage'] == 3:
        print(f"   With offload:    {mem['per_gpu_offload_gb']:.1f} GB")

    print(f"\n🖥️  GPU Recommendations (for {mem['zero_stage']}-stage ZeRO)")
    print(f"   {'GPU':<14} {'#GPUs':<6} {'Util':<6} {'$/hr(ond)':<10} {'$/hr(spot)':<10} {'Total(ond)':<12} {'Total(spot)':<12}")
    print(f"   {'─'*14} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")
    for r in recommendations:
        print(f"   {r['gpu']:<14} {r['n_gpus_min']:<6} {r['mem_utilization']:<6} "
              f"${r['hourly_on_demand']:<9.2f} ${r['hourly_spot']:<9.2f} "
              f"${r['est_total_on_demand']:<11,.0f} ${r['est_total_spot']:<11,.0f}")

    print(f"\n💡 Tips")
    print(f"   1. Activation checkpointing (already enabled) saves ~{mem['activations_gb']*0.5:.0f} GB activations")
    print(f"   2. bf16 is preferred over fp16 on A100/H100 (better numerical range)")
    print(f"   3. Gradient accumulation: increase global batch without memory cost")
    print(f"   4. ZeRO-3 + CPU offload is the most memory-efficient")
    print(f"   5. Spot instances save ~60% (use checkpointing for preemption)")
    print(f"   6. For 200K steps @ 0.5s/step ≈ 28 hours training time")

    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hierarchical MoE memory & cost estimator")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"],
                        help="Model precision")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Micro batch size per GPU")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Sequence length")
    parser.add_argument("--zero-stage", type=int, default=3, choices=[0, 1, 2, 3],
                        help="ZeRO optimization stage")
    args = parser.parse_args()

    spec = ModelSpec()
    mem = estimate_memory_gb(spec, dtype=args.dtype,
                             batch_size=args.batch_size,
                             seq_len=args.seq_len,
                             zero_stage=args.zero_stage)
    recs = recommend_gpus(mem)
    print_report(mem, recs)


if __name__ == "__main__":
    main()
