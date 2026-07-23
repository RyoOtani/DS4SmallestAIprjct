"""Lambda Labs Provider — GPU cloud for deep learning."""
from ..provider import (CloudProvider, InstanceSpec, LaunchConfig, CostEstimate, InstanceStatus, GPUType)
import json, time, urllib.request

class LambdaLabsProvider(CloudProvider):
    NAME = "lambda_labs"
    PRICING = {GPUType.A100_80GB:1.10, GPUType.A100_40GB:0.80, GPUType.H100:2.49, GPUType.A6000:0.50}
    SPOT_DISCOUNT = 0.70
    def __init__(self, api_key: str="", config: dict=None):
        super().__init__(config); self.api_key = api_key or (config or {}).get("lambda_api_key",""); self.api_base = "https://cloud.lambdalabs.com/api/v1"
    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        gpu = spec.gpu_requirement
        payload = {"region_name":"us-west-2","instance_type_name":self._gpu_instance_type(gpu.gpu_type),"ssh_key_names":["tinyllm"],"quantity":1,"name":f"tinyllm-{spec.model_scale}"}
        try:
            data = self._api_post(f"{self.api_base}/instance-operations/launch", payload)
            iid = data.get("data",{}).get("instance_ids",[""])[0]; instance = InstanceSpec(provider=self.NAME, instance_id=iid, instance_name=payload["name"], gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_hour=self.PRICING.get(gpu.gpu_type,1.0)*gpu.count, status=InstanceStatus.PROVISIONING, created_at=time.time()); self._instances[iid] = instance; return instance
        except Exception as e: return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":str(e)})
    def status(self, instance_id: str) -> InstanceStatus:
        try:
            data = self._api_get(f"{self.api_base}/instances/{instance_id}")
            st = data.get("data",{}).get("status",""); return {"active":InstanceStatus.RUNNING,"booting":InstanceStatus.PROVISIONING,"unhealthy":InstanceStatus.ERROR,"terminated":InstanceStatus.TERMINATED}.get(st, InstanceStatus.PROVISIONING)
        except Exception: return InstanceStatus.ERROR
    def list_instances(self) -> list[InstanceSpec]:
        try:
            data = self._api_get(f"{self.api_base}/instances")
            return [InstanceSpec(provider=self.NAME, instance_id=i.get("id",""), instance_name=i.get("name",""), gpu_type=i.get("instance_type",{}).get("name",""), gpu_count=i.get("instance_type",{}).get("specs",{}).get("gpus",1)) for i in data.get("data",[])]
        except Exception: return list(self._instances.values())
    def destroy(self, instance_id: str) -> bool:
        try: self._api_post(f"{self.api_base}/instance-operations/terminate", {"instance_ids":[instance_id]}); self._instances.pop(instance_id,None); return True
        except Exception: return False
    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        gpu = spec.gpu_requirement; ppg = self.PRICING.get(gpu.gpu_type,1.0); pt = ppg*gpu.count; h = spec.auto_shutdown_hours or 24
        return CostEstimate(provider=self.NAME, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_gpu_hour=ppg, price_per_hour_total=pt, estimated_hours=h, estimated_total_cost=pt*h, spot_discount_pct=self.SPOT_DISCOUNT*100, spot_price_total=pt*self.SPOT_DISCOUNT*h)
    def get_available_gpus(self) -> list[dict]: return [{"type":g.value,"price_per_hour":p} for g,p in self.PRICING.items()]
    def _gpu_instance_type(self, gpu: GPUType) -> str:
        return {GPUType.A100_80GB:"gpu_8x_a100_80gb_sxm4",GPUType.A100_40GB:"gpu_8x_a100",GPUType.H100:"gpu_8x_h100",GPUType.A6000:"gpu_1x_a6000"}.get(gpu,"gpu_8x_a100_80gb_sxm4")
    def _api_get(self, e: str) -> dict:
        r = urllib.request.Request(e, headers={"Authorization":f"Bearer {self.api_key}"})
        with urllib.request.urlopen(r, timeout=30) as resp: return json.loads(resp.read())
    def _api_post(self, e: str, d: dict) -> dict:
        r = urllib.request.Request(e, data=json.dumps(d).encode(), headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(r, timeout=30) as resp: return json.loads(resp.read())
