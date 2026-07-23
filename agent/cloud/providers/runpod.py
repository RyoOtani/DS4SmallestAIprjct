"""
RunPod Provider — Serverless & pod-based GPU cloud.

API: https://docs.runpod.io/
Price: ~$1.89/hr A100-80GB (spot: ~$0.99/hr)

Key features:
  ✅ Pod-based (stateful) and Serverless (stateless)
  ✅ Template-based launch
  ✅ Auto-shutdown via idle timeout
  ✅ Network storage volumes
"""

from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from ..provider import (
    CloudProvider, InstanceSpec, LaunchConfig, CostEstimate,
    InstanceStatus, GPUType, GPURequirement,
)


class RunPodProvider(CloudProvider):
    """RunPod GPU cloud provider."""

    NAME = "runpod"

    # Pricing (USD/hr, on-demand)
    PRICING = {
        GPUType.A100_80GB:   1.89,
        GPUType.A100_40GB:   1.49,
        GPUType.H100:        2.99,
        GPUType.L40S:        0.99,
        GPUType.L4:          0.44,
        GPUType.T4:          0.24,
        GPUType.V100:        0.54,
        GPUType.RTX4090:     0.79,
        GPUType.RTX3090:     0.39,
        GPUType.A6000:       0.69,
    }

    SPOT_DISCOUNT = 0.48  # ~48-52% off for spot

    def __init__(self, api_key: str = "", config: dict = None):
        super().__init__(config)
        self.api_key = api_key or (config or {}).get("runpod_api_key", "")
        self.api_base = "https://api.runpod.io/v2"
        self.rest_base = "https://rest.runpod.io/v1"

    # ── Launch ────────────────────────────────────────────────────────────

    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        """Launch a RunPod GPU pod."""
        gpu = spec.gpu_requirement

        payload = {
            "name": f"tinyllm-{spec.model_scale}-{int(time.time())}",
            "imageName": spec.docker_image,
            "gpuTypeId": self._gpu_type_id(gpu.gpu_type),
            "gpuCount": gpu.count,
            "containerDiskInGb": 200,
            "minVramGb": gpu.min_vram_per_gpu_gb,
            "volumeInGb": 500,
            "volumeMountPath": "/workspace",
            "env": [
                {"key": k, "value": v}
                for k, v in (spec.env_vars or {}).items()
            ],
            "startScript": spec.startup_script or self.generate_startup_script(spec),
            "bidPerGpu": spec.max_price_per_hour / gpu.count if spec.use_spot else 0,
        }

        # Add provider-specific overrides
        payload.update(spec.provider_config.get("runpod", {}))

        try:
            data = self._api_post(f"{self.rest_base}/pods", payload)
            instance = InstanceSpec(
                provider=self.NAME,
                instance_id=data.get("id", ""),
                instance_name=payload["name"],
                gpu_type=gpu.gpu_type.value,
                gpu_count=gpu.count,
                price_per_hour=self.PRICING.get(gpu.gpu_type, 2.0) * gpu.count,
                status=InstanceStatus.PROVISIONING,
                created_at=time.time(),
                metadata={"pod_id": data.get("id")},
            )
            self._instances[instance.instance_id] = instance
            return instance
        except Exception as e:
            return InstanceSpec(
                provider=self.NAME,
                status=InstanceStatus.ERROR,
                metadata={"error": str(e)},
            )

    def status(self, instance_id: str) -> InstanceStatus:
        """Get RunPod pod status."""
        try:
            data = self._api_get(f"{self.rest_base}/pods/{instance_id}")
            state = data.get("desiredStatus", "").upper()
            status_map = {
                "RUNNING": InstanceStatus.RUNNING,
                "CREATED": InstanceStatus.PROVISIONING,
                "INITIALIZING": InstanceStatus.PROVISIONING,
                "STOPPED": InstanceStatus.STOPPED,
                "EXITED": InstanceStatus.TERMINATED,
            }
            return status_map.get(state, InstanceStatus.PROVISIONING)
        except Exception:
            return InstanceStatus.ERROR

    def list_instances(self) -> list[InstanceSpec]:
        """List all RunPod pods."""
        try:
            data = self._api_get(f"{self.rest_base}/pods")
            pods = data if isinstance(data, list) else data.get("data", [])
            return [
                InstanceSpec(
                    provider=self.NAME,
                    instance_id=p.get("id", ""),
                    instance_name=p.get("name", ""),
                    gpu_type=p.get("gpuTypeId", ""),
                    gpu_count=p.get("gpuCount", 1),
                    status=self._parse_status(p.get("desiredStatus", "")),
                )
                for p in pods
            ]
        except Exception:
            return list(self._instances.values())

    def destroy(self, instance_id: str) -> bool:
        """Terminate a RunPod pod."""
        try:
            self._api_delete(f"{self.rest_base}/pods/{instance_id}")
            self._instances.pop(instance_id, None)
            return True
        except Exception:
            return False

    # ── Cost ──────────────────────────────────────────────────────────────

    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        """Estimate RunPod cost."""
        gpu = spec.gpu_requirement
        price_per_gpu = self.PRICING.get(gpu.gpu_type, 2.0)
        price_total = price_per_gpu * gpu.count
        estimated_hours = spec.auto_shutdown_hours or 24

        return CostEstimate(
            provider=self.NAME,
            gpu_type=gpu.gpu_type.value,
            gpu_count=gpu.count,
            price_per_gpu_hour=price_per_gpu,
            price_per_hour_total=price_total,
            estimated_hours=estimated_hours,
            estimated_total_cost=price_total * estimated_hours,
            spot_discount_pct=self.SPOT_DISCOUNT * 100,
            spot_price_total=price_total * self.SPOT_DISCOUNT * estimated_hours,
        )

    def get_available_gpus(self) -> list[dict]:
        """List available RunPod GPU types."""
        return [
            {"type": gpu.value, "price_per_hour": price,
             "spot_price": round(price * self.SPOT_DISCOUNT, 2)}
            for gpu, price in self.PRICING.items()
        ]

    # ── Helpers ──────────────────────────────────────────────────────────

    def _gpu_type_id(self, gpu: GPUType) -> str:
        """Map GPUType to RunPod GPU type ID."""
        mapping = {
            GPUType.A100_80GB: "NVIDIA A100 80GB PCIe",
            GPUType.A100_40GB: "NVIDIA A100-SXM4-40GB",
            GPUType.H100: "NVIDIA H100 PCIe",
            GPUType.L40S: "NVIDIA L40S",
            GPUType.L4: "NVIDIA L4",
            GPUType.T4: "NVIDIA T4",
            GPUType.V100: "NVIDIA V100",
            GPUType.RTX4090: "NVIDIA GeForce RTX 4090",
            GPUType.RTX3090: "NVIDIA GeForce RTX 3090",
            GPUType.A6000: "NVIDIA RTX A6000",
        }
        return mapping.get(gpu, "NVIDIA A100 80GB PCIe")

    def _parse_status(self, state: str) -> InstanceStatus:
        return {
            "RUNNING": InstanceStatus.RUNNING,
            "CREATED": InstanceStatus.PROVISIONING,
            "EXITED": InstanceStatus.TERMINATED,
        }.get(state.upper(), InstanceStatus.PROVISIONING)

    def _api_get(self, endpoint: str) -> dict:
        req = urllib.request.Request(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _api_post(self, endpoint: str, data: dict) -> dict:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(data).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _api_delete(self, endpoint: str) -> dict:
        req = urllib.request.Request(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
