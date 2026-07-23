"""GPU Soroban Provider — Japanese GPU cloud with competitive pricing."""
from __future__ import annotations
import json, time, urllib.request
from typing import Optional
from ..provider import (CloudProvider, InstanceSpec, LaunchConfig, CostEstimate, InstanceStatus, GPUType)

class GPUSorobanProvider(CloudProvider):
    """GPU Soroban (GPUSOROBAN) — Japan-based GPU cloud."""
    NAME = "gpu_soroban"
    PRICING = {GPUType.A100_80GB:2.10, GPUType.A100_40GB:1.70, GPUType.H100:3.20, GPUType.L40S:1.10, GPUType.V100:0.70}
    SPOT_DISCOUNT = 0.60
    def __init__(self, api_key: str="", config: dict=None):
        super().__init__(config); self.api_key = api_key or (config or {}).get("soroban_api_key",""); self.api_base = "https://api.gp-soroban.com/v1"
    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        gpu = spec.gpu_requirement
        payload = {"gpu_type": gpu.gpu_type.value, "gpu_count": gpu.count, "disk_gb": 500, "image": spec.docker_image, "startup_script": spec.startup_script or self.generate_startup_script(spec), "spot": spec.use_spot}
        try:
            data = self._api_post(f"{self.api_base}/instances", payload)
            instance = InstanceSpec(provider=self.NAME, instance_id=data.get("instance_id",""), instance_name=f"tinyllm-{spec.model_scale}", gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_hour=self.PRICING.get(gpu.gpu_type,2.0)*gpu.count, region="jp", status=InstanceStatus.PROVISIONING, created_at=time.time())
            self._instances[instance.instance_id] = instance; return instance
        except Exception as e: return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":str(e)})
    def status(self, instance_id: str) -> InstanceStatus:
        try:
            data = self._api_get(f"{self.api_base}/instances/{instance_id}")
            return {"running":InstanceStatus.RUNNING,"creating":InstanceStatus.PROVISIONING,"stopped":InstanceStatus.STOPPED,"terminated":InstanceStatus.TERMINATED}.get(data.get("status",""), InstanceStatus.PROVISIONING)
        except Exception: return InstanceStatus.ERROR
    def list_instances(self) -> list[InstanceSpec]:
        try:
            data = self._api_get(f"{self.api_base}/instances")
            return [InstanceSpec(provider=self.NAME, instance_id=i.get("instance_id",""), instance_name=i.get("name",""), gpu_type=i.get("gpu_type",""), gpu_count=i.get("gpu_count",1), status=InstanceStatus.RUNNING) for i in data.get("instances",[])]
        except Exception: return list(self._instances.values())
    def destroy(self, instance_id: str) -> bool:
        try: self._api_delete(f"{self.api_base}/instances/{instance_id}"); self._instances.pop(instance_id,None); return True
        except Exception: return False
    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        gpu = spec.gpu_requirement; ppg = self.PRICING.get(gpu.gpu_type,2.0); pt = ppg*gpu.count; h = spec.auto_shutdown_hours or 24
        return CostEstimate(provider=self.NAME, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_gpu_hour=ppg, price_per_hour_total=pt, estimated_hours=h, estimated_total_cost=pt*h, spot_discount_pct=self.SPOT_DISCOUNT*100, spot_price_total=pt*self.SPOT_DISCOUNT*h)
    def get_available_gpus(self) -> list[dict]: return [{"type":g.value,"price_per_hour":p,"spot_price":round(p*self.SPOT_DISCOUNT,2)} for g,p in self.PRICING.items()]
    def _api_get(self, e: str) -> dict:
        r = urllib.request.Request(e, headers={"Authorization":f"Bearer {self.api_key}"})
        with urllib.request.urlopen(r, timeout=30) as resp: return json.loads(resp.read())
    def _api_post(self, e: str, d: dict) -> dict:
        r = urllib.request.Request(e, data=json.dumps(d).encode(), headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(r, timeout=30) as resp: return json.loads(resp.read())
    def _api_delete(self, e: str) -> dict:
        r = urllib.request.Request(e, headers={"Authorization":f"Bearer {self.api_key}"}, method="DELETE")
        with urllib.request.urlopen(r, timeout=30) as resp: return json.loads(resp.read())
