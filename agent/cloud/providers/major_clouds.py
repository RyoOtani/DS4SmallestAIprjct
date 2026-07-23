"""AWS, Azure, GCP Providers — Major cloud GPU providers via CLI wrappers."""
from __future__ import annotations
import json, subprocess, time
from typing import Optional
from ..provider import (CloudProvider, InstanceSpec, LaunchConfig, CostEstimate, InstanceStatus, GPUType)

# ══════════════════════════════════════════════════════════════════════════════
# AWS Provider
# ══════════════════════════════════════════════════════════════════════════════

class AWSProvider(CloudProvider):
    """AWS EC2 GPU instances (p4d/p5/g5/g6). Uses `aws` CLI."""
    NAME = "aws"
    PRICING = {GPUType.A100_80GB: 32.77/8, GPUType.H100: 98.32/8, GPUType.L4: 0.75, GPUType.T4: 0.53, GPUType.V100: 3.06, GPUType.A6000: 1.50}
    SPOT_DISCOUNT = 0.30

    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        gpu = spec.gpu_requirement
        instance_type = self._instance_type(gpu)
        cmd = ["aws","ec2","run-instances","--instance-type",instance_type,"--image-id",self.config.get("aws_ami","ami-0abcdef1234567890"),"--count","1","--key-name",self.config.get("aws_key","tinyllm"),"--security-group-ids",self.config.get("aws_sg","sg-default"),"--block-device-mappings",f"DeviceName=/dev/sda1,Ebs={{VolumeSize=500}}"]
        if spec.use_spot: cmd += ["--instance-market-options",'{"MarketType":"spot"}']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            data = json.loads(result.stdout)
            iid = data["Instances"][0]["InstanceId"]
            instance = InstanceSpec(provider=self.NAME, instance_id=iid, instance_name=f"tinyllm-{spec.model_scale}", gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, status=InstanceStatus.PROVISIONING, created_at=time.time())
            self._instances[iid] = instance; return instance
        except Exception as e: return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":str(e)})

    def status(self, instance_id: str) -> InstanceStatus:
        try:
            r = subprocess.run(["aws","ec2","describe-instances","--instance-ids",instance_id], capture_output=True, text=True, timeout=30)
            state = json.loads(r.stdout)["Reservations"][0]["Instances"][0]["State"]["Name"]
            return {"running":InstanceStatus.RUNNING,"pending":InstanceStatus.PROVISIONING,"stopped":InstanceStatus.STOPPED,"terminated":InstanceStatus.TERMINATED}.get(state, InstanceStatus.PROVISIONING)
        except Exception: return InstanceStatus.ERROR

    def list_instances(self) -> list[InstanceSpec]:
        try:
            r = subprocess.run(["aws","ec2","describe-instances","--filters","Name=tag:Name,Values=tinyllm-*"], capture_output=True, text=True, timeout=30)
            instances = [i for r in json.loads(r.stdout).get("Reservations",[]) for i in r.get("Instances",[])]
            return [InstanceSpec(provider=self.NAME, instance_id=i["InstanceId"], instance_name=next((t["Value"] for t in i.get("Tags",[]) if t["Key"]=="Name"),""), status=InstanceStatus.RUNNING) for i in instances]
        except Exception: return list(self._instances.values())

    def destroy(self, instance_id: str) -> bool:
        try: subprocess.run(["aws","ec2","terminate-instances","--instance-ids",instance_id], capture_output=True, timeout=60); self._instances.pop(instance_id,None); return True
        except Exception: return False

    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        gpu = spec.gpu_requirement; ppg = self.PRICING.get(gpu.gpu_type,4.0); pt = ppg*gpu.count; h = spec.auto_shutdown_hours or 24
        return CostEstimate(provider=self.NAME, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_gpu_hour=ppg, price_per_hour_total=pt, estimated_hours=h, estimated_total_cost=pt*h, spot_discount_pct=self.SPOT_DISCOUNT*100, spot_price_total=pt*self.SPOT_DISCOUNT*h)

    def get_available_gpus(self) -> list[dict]: return [{"type":g.value,"price_per_hour":p} for g,p in self.PRICING.items()]
    def _instance_type(self, gpu: GPUType) -> str:
        return {GPUType.A100_80GB:"p4d.24xlarge",GPUType.H100:"p5.48xlarge",GPUType.L4:"g6.xlarge",GPUType.T4:"g4dn.xlarge",GPUType.V100:"p3.2xlarge"}.get(gpu,"p4d.24xlarge")


# ══════════════════════════════════════════════════════════════════════════════
# Azure Provider
# ══════════════════════════════════════════════════════════════════════════════

class AzureProvider(CloudProvider):
    """Azure NC-series GPU VMs. Uses `az` CLI."""
    NAME = "azure"
    PRICING = {GPUType.A100_80GB: 3.67, GPUType.H100: 6.98, GPUType.T4: 0.55, GPUType.V100: 2.48}
    SPOT_DISCOUNT = 0.20

    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        gpu = spec.gpu_requirement; vm_size = self._vm_size(gpu)
        rg = self.config.get("azure_rg","tinyllm-rg"); name = f"tinyllm-{spec.model_scale}-{int(time.time())}"
        cmd = ["az","vm","create","--resource-group",rg,"--name",name,"--image","Ubuntu2204","--size",vm_size,"--admin-username","tinyllm","--generate-ssh-keys"]
        if spec.use_spot: cmd += ["--priority","Spot","--max-price",str(spec.max_price_per_hour)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120); instance = InstanceSpec(provider=self.NAME, instance_id=name, instance_name=name, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, status=InstanceStatus.PROVISIONING, created_at=time.time()); self._instances[name] = instance; return instance
        except Exception as e: return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":str(e)})

    def status(self, instance_id: str) -> InstanceStatus:
        try:
            r = subprocess.run(["az","vm","show","--resource-group",self.config.get("azure_rg","tinyllm-rg"),"--name",instance_id,"--query","powerState","-o","tsv"], capture_output=True, text=True, timeout=30)
            return {"VM running":InstanceStatus.RUNNING,"VM starting":InstanceStatus.PROVISIONING,"VM stopped":InstanceStatus.STOPPED,"VM deallocated":InstanceStatus.TERMINATED}.get(r.stdout.strip(), InstanceStatus.PROVISIONING)
        except Exception: return InstanceStatus.ERROR

    def list_instances(self) -> list[InstanceSpec]:
        try:
            r = subprocess.run(["az","vm","list","--resource-group",self.config.get("azure_rg","tinyllm-rg"),"--query","[].{name:name,location:location}"], capture_output=True, text=True, timeout=30)
            return [InstanceSpec(provider=self.NAME, instance_id=v.get("name",""), instance_name=v.get("name",""), region=v.get("location","")) for v in json.loads(r.stdout)]
        except Exception: return list(self._instances.values())

    def destroy(self, instance_id: str) -> bool:
        try: subprocess.run(["az","vm","delete","--resource-group",self.config.get("azure_rg","tinyllm-rg"),"--name",instance_id,"--yes"], capture_output=True, timeout=60); self._instances.pop(instance_id,None); return True
        except Exception: return False

    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        gpu = spec.gpu_requirement; ppg = self.PRICING.get(gpu.gpu_type,4.0); pt = ppg*gpu.count; h = spec.auto_shutdown_hours or 24
        return CostEstimate(provider=self.NAME, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_gpu_hour=ppg, price_per_hour_total=pt, estimated_hours=h, estimated_total_cost=pt*h, spot_discount_pct=self.SPOT_DISCOUNT*100, spot_price_total=pt*self.SPOT_DISCOUNT*h)

    def get_available_gpus(self) -> list[dict]: return [{"type":g.value,"price_per_hour":p} for g,p in self.PRICING.items()]
    def _vm_size(self, gpu: GPUType) -> str:
        return {GPUType.A100_80GB:"Standard_NC96ads_A100_v4",GPUType.H100:"Standard_NC40ads_H100_v5",GPUType.T4:"Standard_NC4as_T4_v3",GPUType.V100:"Standard_NC6s_v3"}.get(gpu,"Standard_NC96ads_A100_v4")


# ══════════════════════════════════════════════════════════════════════════════
# GCP Provider
# ══════════════════════════════════════════════════════════════════════════════

class GCPProvider(CloudProvider):
    """Google Cloud GPU VMs. Uses `gcloud` CLI."""
    NAME = "gcp"
    PRICING = {GPUType.A100_80GB: 3.67, GPUType.H100: 5.78, GPUType.L4: 0.55, GPUType.T4: 0.35, GPUType.V100: 2.48}
    SPOT_DISCOUNT = 0.30

    def launch(self, spec: LaunchConfig) -> InstanceSpec:
        gpu = spec.gpu_requirement; gpu_str = self._gpu_str(gpu); name = f"tinyllm-{spec.model_scale}-{int(time.time())}"
        cmd = ["gcloud","compute","instances","create",name,"--machine-type",self._machine_type(gpu),"--zone",self.config.get("gcp_zone","us-central1-a"),"--image-family","ubuntu-2204-lts","--image-project","ubuntu-os-cloud","--boot-disk-size","500GB","--accelerator",f"type={gpu_str},count={gpu.count}","--maintenance-policy","TERMINATE"]
        if spec.use_spot: cmd += ["--provisioning-model","SPOT"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120); instance = InstanceSpec(provider=self.NAME, instance_id=name, instance_name=name, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, status=InstanceStatus.PROVISIONING, created_at=time.time()); self._instances[name] = instance; return instance
        except Exception as e: return InstanceSpec(provider=self.NAME, status=InstanceStatus.ERROR, metadata={"error":str(e)})

    def status(self, instance_id: str) -> InstanceStatus:
        try:
            r = subprocess.run(["gcloud","compute","instances","describe",instance_id,"--zone",self.config.get("gcp_zone","us-central1-a"),"--format=value(status)"], capture_output=True, text=True, timeout=30)
            return {"RUNNING":InstanceStatus.RUNNING,"PROVISIONING":InstanceStatus.PROVISIONING,"STOPPING":InstanceStatus.STOPPED,"TERMINATED":InstanceStatus.TERMINATED}.get(r.stdout.strip(), InstanceStatus.PROVISIONING)
        except Exception: return InstanceStatus.ERROR

    def list_instances(self) -> list[InstanceSpec]:
        try:
            r = subprocess.run(["gcloud","compute","instances","list","--filter=name~tinyllm","--format=json(name,zone,status)"], capture_output=True, text=True, timeout=30)
            return [InstanceSpec(provider=self.NAME, instance_id=i.get("name",""), instance_name=i.get("name",""), region=i.get("zone","").rsplit("-",1)[0]) for i in json.loads(r.stdout)]
        except Exception: return list(self._instances.values())

    def destroy(self, instance_id: str) -> bool:
        try: subprocess.run(["gcloud","compute","instances","delete",instance_id,"--zone",self.config.get("gcp_zone","us-central1-a"),"--quiet"], capture_output=True, timeout=60); self._instances.pop(instance_id,None); return True
        except Exception: return False

    def estimate_cost(self, spec: LaunchConfig) -> CostEstimate:
        gpu = spec.gpu_requirement; ppg = self.PRICING.get(gpu.gpu_type,3.0); pt = ppg*gpu.count; h = spec.auto_shutdown_hours or 24
        return CostEstimate(provider=self.NAME, gpu_type=gpu.gpu_type.value, gpu_count=gpu.count, price_per_gpu_hour=ppg, price_per_hour_total=pt, estimated_hours=h, estimated_total_cost=pt*h, spot_discount_pct=self.SPOT_DISCOUNT*100, spot_price_total=pt*self.SPOT_DISCOUNT*h)

    def get_available_gpus(self) -> list[dict]: return [{"type":g.value,"price_per_hour":p} for g,p in self.PRICING.items()]
    def _gpu_str(self, gpu: GPUType) -> str:
        return {GPUType.A100_80GB:"nvidia-a100-80gb",GPUType.H100:"nvidia-h100-80gb",GPUType.L4:"nvidia-l4",GPUType.T4:"nvidia-tesla-t4",GPUType.V100:"nvidia-tesla-v100"}.get(gpu,"nvidia-a100-80gb")
    def _machine_type(self, gpu: GPUType) -> str:
        return {GPUType.A100_80GB:"a2-highgpu-8g",GPUType.H100:"a3-highgpu-8g",GPUType.L4:"g2-standard-4",GPUType.T4:"n1-standard-8",GPUType.V100:"n1-standard-8"}.get(gpu,"a2-highgpu-8g")
