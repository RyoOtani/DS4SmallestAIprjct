"""
Cloud Provider Abstraction Layer — GPU-cloud agnostic training/inference.

Design:
  CloudProvider (abstract base)
    ├── RunPodProvider
    ├── VastAIProvider
    ├── GPUSorobanProvider
    ├── LambdaLabsProvider
    ├── AWSProvider
    ├── AzureProvider
    └── GCPProvider

Usage:
  tinyllm cloud launch --provider runpod --gpus 8 --config xlarge
  tinyllm cloud launch --provider vast    --gpus 4 --config medium --spot
  tinyllm cloud status  --provider aws
  tinyllm cloud destroy --provider runpod --instance i-xxxx
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


# ── Shared Types ─────────────────────────────────────────────────────────────

class InstanceStatus(Enum):
    PROVISIONING = "provisioning"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


class GPUType(Enum):
    """Common GPU types across providers."""
    A100_40GB = "a100-40gb"
    A100_80GB = "a100-80gb"
    H100 = "h100"
    L40S = "l40s"
    L4 = "l4"
    T4 = "t4"
    V100 = "v100"
    RTX4090 = "rtx4090"
    RTX3090 = "rtx3090"
    A6000 = "a6000"


@dataclass
class GPURequirement:
    """GPU requirements for a training job."""
    gpu_type: GPUType = GPUType.A100_80GB
    count: int = 8
    memory_gb: int = 80
    min_vram_per_gpu_gb: int = 40
    interconnect: str = "nvlink"  # nvlink, pcie, infiniband


@dataclass
class InstanceSpec:
    """Specification for a cloud GPU instance."""
    provider: str
    instance_id: str = ""
    instance_name: str = ""
    gpu_type: str = ""
    gpu_count: int = 1
    cpu_cores: int = 16
    memory_gb: int = 128
    disk_gb: int = 200
    price_per_hour: float = 0.0
    region: str = ""
    status: InstanceStatus = InstanceStatus.PROVISIONING
    ssh_host: str = ""
    ssh_port: int = 22
    ssh_key_path: str = ""
    docker_image: str = ""
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class LaunchConfig:
    """Configuration for launching a training job."""
    # Model
    model_scale: str = "medium"      # nano/small/medium/large/xlarge/...
    model_config_path: str = ""

    # GPU
    gpu_requirement: GPURequirement = field(default_factory=GPURequirement)
    use_spot: bool = True             # Spot/preemptible (cheaper)
    max_price_per_hour: float = 50.0  # Budget cap

    # Training
    training_command: str = ""        # Override default
    dataset_path: str = ""            # S3/GS/HTTP path to dataset
    checkpoint_path: str = ""         # Where to save checkpoints
    num_epochs: int = 3
    batch_size: int = 0              # 0 = auto

    # Environment
    docker_image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
    env_vars: dict[str, str] = field(default_factory=dict)
    startup_script: str = ""          # Bash script to run on boot

    # Lifecycle
    auto_shutdown_hours: float = 24.0  # Auto-terminate after N hours
    keep_alive: bool = False           # Don't auto-shutdown

    # Provider-specific overrides
    provider_config: dict = field(default_factory=dict)


@dataclass
class CostEstimate:
    """Estimated cost for a training run."""
    provider: str
    gpu_type: str
    gpu_count: int
    price_per_gpu_hour: float
    price_per_hour_total: float
    estimated_hours: float
    estimated_total_cost: float
    spot_discount_pct: float = 50.0
    spot_price_total: float = 0.0
    currency: str = "USD"

    def summary(self) -> str:
        return (
            f"💰 {self.provider}: {self.gpu_count}x {self.gpu_type} @ "
            f"${self.price_per_hour_total:.2f}/hr → "
            f"~${self.estimated_total_cost:.0f} total "
            f"(${self.spot_price_total:.0f} spot, {self.estimated_hours:.0f}h)"
        )


# ── Abstract Base ────────────────────────────────────────────────────────────

class CloudProvider(ABC):
    """
    Abstract base for all cloud GPU providers.

    Every provider MUST implement:
      - launch(): Create GPU instance(s)
      - status(): Get instance status
      - list_instances(): List all instances
      - destroy(): Terminate instance(s)
      - estimate_cost(): Estimate training cost
      - get_available_gpus(): List available GPU types
    """

    NAME: str = "base"

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._instances: dict[str, InstanceSpec] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abstractmethod
    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        """Launch a GPU instance with the given configuration."""
        ...

    @abstractmethod
    def status(self, instance_id: str) -> InstanceStatus:
        """Get the current status of an instance."""
        ...

    @abstractmethod
    def list_instances(self) -> list[InstanceSpec]:
        """List all instances managed by this provider."""
        ...

    @abstractmethod
    def destroy(self, instance_id: str) -> bool:
        """Terminate/destroy an instance."""
        ...

    # ── Cost ──────────────────────────────────────────────────────────────

    @abstractmethod
    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        """Estimate the cost of a training run."""
        ...

    @abstractmethod
    def get_available_gpus(self) -> list[dict]:
        """List available GPU types and their prices."""
        ...

    # ── Utility ───────────────────────────────────────────────────────────

    def wait_until_ready(
        self,
        instance_id: str,
        timeout_s: int = 600,
        on_status: Callable[[InstanceStatus], None] = None,
    ) -> bool:
        """Poll until instance is RUNNING or timeout."""
        import time
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st = self.status(instance_id)
            if on_status:
                on_status(st)
            if st == InstanceStatus.RUNNING:
                return True
            if st == InstanceStatus.ERROR:
                return False
            time.sleep(10)
        return False

    def generate_startup_script(self, config: LaunchConfig) -> str:
        """Generate a startup script for training."""
        lines = [
            "#!/bin/bash",
            "set -e",
            "",
            "# TinyLLM Cloud Training Startup",
            f"echo 'Starting TinyLLM training: {config.model_scale}'",
            "",
            "# Clone repo",
            "if [ ! -d DS4SmallestAIprjct ]; then",
            "  git clone https://github.com/RyoOtani/DS4SmallestAIprjct.git",
            "fi",
            "cd DS4SmallestAIprjct",
            "",
            "# Install deps",
            "pip install -r requirements.txt -q",
            "",
            "# Download dataset (if specified)",
        ]

        if config.dataset_path:
            if config.dataset_path.startswith("s3://"):
                lines.append(f"aws s3 sync {config.dataset_path} /data/")
            elif config.dataset_path.startswith("gs://"):
                lines.append(f"gsutil -m rsync {config.dataset_path} /data/")
            else:
                lines.append(f"wget -q {config.dataset_path} -P /data/")

        lines.append("")
        lines.append("# Launch training")
        cmd = config.training_command or (
            f"torchrun --nproc_per_node={config.gpu_requirement.count} "
            f"-m agent.phase7.cli train --config {config.model_scale}"
        )
        lines.append(cmd)

        lines.append("")
        lines.append("# Auto-shutdown (if not keep-alive)")
        if not config.keep_alive:
            hours = int(config.auto_shutdown_hours * 3600)
            lines.append(f"echo 'Will auto-shutdown in {config.auto_shutdown_hours}h'")
            lines.append(f"(sleep {hours} && sudo shutdown -h now) &")

        return "\n".join(lines)
