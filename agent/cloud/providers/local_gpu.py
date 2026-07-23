"""Local GPU Provider — Metal / CUDA / Vulkan for on-device training/inference."""
from __future__ import annotations
import platform, subprocess, time, json
from typing import Optional
from ..provider import (CloudProvider, InstanceSpec, LaunchConfig, CostEstimate, InstanceStatus, GPUType)

class LocalGPUProvider(CloudProvider):
    """Local GPU provider — uses the machine's built-in GPU (Metal/CUDA/Vulkan)."""
    NAME = "local"

    def __init__(self, config: dict=None):
        super().__init__(config)
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect local GPU capabilities."""
        self._gpu_name = "CPU"
        self._gpu_count = 0
        self._backend = "cpu"
        system = platform.system()

        if system == "Darwin":
            # Check for Apple Silicon GPU via Metal
            try:
                import subprocess
                r = subprocess.run(["system_profiler","SPDisplaysDataType"], capture_output=True, text=True, timeout=10)
                for line in r.stdout.split("\n"):
                    if "Chipset Model" in line:
                        self._gpu_name = line.split(":")[-1].strip()
                    if "Total Number of Cores" in line:
                        self._gpu_count = int(line.split(":")[-1].strip())
                if "Apple M" in self._gpu_name or "Apple" in self._gpu_name:
                    self._backend = "metal"
                    if self._gpu_count == 0:
                        self._gpu_count = 1  # Unified memory GPU
            except Exception:
                self._gpu_name = "Apple Silicon (Metal)"
                self._gpu_count = 1
                self._backend = "metal"

        elif system == "Linux":
            try:
                r = subprocess.run(["nvidia-smi","--query-gpu=name,count","--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                self._gpu_name = r.stdout.strip().split(",")[0] if r.stdout else "NVIDIA GPU"
                self._gpu_count = int(r.stdout.strip().split(",")[1]) if "," in r.stdout else 1
                self._backend = "cuda"
            except Exception:
                self._gpu_name = "Unknown GPU"
                self._backend = "vulkan"

        elif system == "Windows":
            self._gpu_name = "Windows GPU"
            self._backend = "directx"

    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        """Launch 'locally' — just validate and return the local GPU info."""
        return InstanceSpec(
            provider=self.NAME,
            instance_id="local",
            instance_name=f"local-{self._gpu_name}",
            gpu_type=self._backend,
            gpu_count=self._gpu_count,
            price_per_hour=0.0,  # Free! (minus electricity)
            status=InstanceStatus.RUNNING,
            created_at=time.time(),
            ssh_host="localhost",
            metadata={"backend": self._backend, "gpu_name": self._gpu_name},
        )

    def status(self, instance_id: str) -> InstanceStatus:
        return InstanceStatus.RUNNING

    def list_instances(self) -> list[InstanceSpec]:
        return [InstanceSpec(provider=self.NAME, instance_id="local", instance_name=self._gpu_name, gpu_type=self._backend, gpu_count=self._gpu_count, status=InstanceStatus.RUNNING, price_per_hour=0.0)]

    def destroy(self, instance_id: str) -> bool:
        # Can't destroy local GPU — just return True
        return True

    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        return CostEstimate(provider="local", gpu_type=self._backend, gpu_count=self._gpu_count, price_per_gpu_hour=0.0, price_per_hour_total=0.0, estimated_hours=spec.auto_shutdown_hours or 24, estimated_total_cost=0.0, spot_discount_pct=0, spot_price_total=0.0, currency="USD (free)")

    def get_available_gpus(self) -> list[dict]:
        return [{"type": self._backend, "name": self._gpu_name, "count": self._gpu_count, "price_per_hour": 0.0, "note": "Local GPU — no cloud cost"}]

    def get_gpu_info(self) -> dict:
        return {"backend": self._backend, "name": self._gpu_name, "count": self._gpu_count}
