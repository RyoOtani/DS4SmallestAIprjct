"""Vast.ai Provider — GPU marketplace with competitive spot pricing."""
from __future__ import annotations
import json, time, urllib.request
from typing import Optional
from ..provider import (
    CloudProvider, InstanceSpec, LaunchConfig, CostEstimate,
    InstanceStatus, GPUType,
)

class VastAIProvider(CloudProvider):
    NAME = "vastai"
    PRICING = {GPUType.A100_80GB:1.20, GPUType.A100_40GB:0.80, GPUType.H100:2.50, GPUType.L40S:0.70, GPUType.RTX4090:0.50, GPUType.RTX3090:0.30, GPUType.A6000:0.45}
    SPOT_DISCOUNT = 0.65
    def __init__(self, api_key: str="", config: dict=None):
        super().__init__(config); self.api_key = api_key or (config or {}).get("vastai_api_key",""); self.api_base = "https://console.vast.ai/api/v0"
    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        gpu = spec.gpu_requirement
        query = {"query": {"gpu_name": gpu.gpu_type.value.replace("-"," ").upper(), "num_gpus": gpu.count, "min_gpu_ram": gpu.min_vram_per_gpu_gb * 1024, "order": [["score","desc"]], "type": "on-demand" if not spec.use_spot else "interruptible"}}
        try:
            offers = self._api_post(f"{self.api_base}/search/offers/", query)
            if not offers.get("offers"):
                return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":"No matching offers"})
            best = offers["offers"][0]
            instance = InstanceSpec(provider=self.NAME, instance_id=str(best.get("id","")), instance_name=f"tinyllm-{spec.model_scale}", gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_hour=best.get("dph_total",0)/100, status=InstanceStatus.PROVISIONING, created_at=time.time(), ssh_host=best.get("ssh_host",""), ssh_port=best.get("ssh_port",22), metadata={"offer":best})
            self._instances[instance.instance_id] = instance; return instance
        except Exception as e: return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":str(e)})
    def status(self, instance_id: str) -> InstanceStatus:
        try:
            data = self._api_get(f"{self.api_base}/instances/"); instances = data.get("instances",[])
            for inst in instances:
                if str(inst.get("id")) == instance_id:
                    return {"running":InstanceStatus.RUNNING,"loading":InstanceStatus.PROVISIONING,"stopped":InstanceStatus.STOPPED}.get(inst.get("cur_state",""), InstanceStatus.PROVISIONING)
            return InstanceStatus.TERMINATED
        except Exception: return InstanceStatus.ERROR
    def list_instances(self) -> list[InstanceSpec]:
        try:
            data = self._api_get(f"{self.api_base}/instances/")
            return [InstanceSpec(provider=self.NAME, instance_id=str(i.get("id","")), instance_name=i.get("label",""), gpu_type=i.get("gpu_name",""), gpu_count=i.get("num_gpus",1)) for i in data.get("instances",[])]
        except Exception: return list(self._instances.values())
    def destroy(self, instance_id: str) -> bool:
        try: self._api_delete(f"{self.api_base}/instances/{instance_id}/"); self._instances.pop(instance_id,None); return True
        except Exception: return False
    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        gpu = spec.gpu_requirement; ppg = self.PRICING.get(gpu.gpu_type,1.0); pt = ppg*gpu.count; h = spec.auto_shutdown_hours or 24
        return CostEstimate(provider=self.NAME, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_gpu_hour=ppg, price_per_hour_total=pt, estimated_hours=h, estimated_total_cost=pt*h, spot_discount_pct=self.SPOT_DISCOUNT*100, spot_price_total=pt*self.SPOT_DISCOUNT*h)
    def get_available_gpus(self) -> list[dict]: return [{"type":g.value,"price_per_hour":p,"spot_price":round(p*self.SPOT_DISCOUNT,2)} for g,p in self.PRICING.items()]
    def _api_get(self, endpoint: str) -> dict:
        req = urllib.request.Request(endpoint, headers={"Authorization":f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
    def _api_post(self, endpoint: str, data: dict) -> dict:
        req = urllib.request.Request(endpoint, data=json.dumps(data).encode(), headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
    def _api_delete(self, endpoint: str) -> dict:
        req = urllib.request.Request(endpoint, headers={"Authorization":f"Bearer {self.api_key}"}, method="DELETE")
        with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
