"""Tests for Cloud GPU Abstraction Layer."""
from __future__ import annotations
import sys, tempfile, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.cloud.provider import (
    CloudProvider, LaunchConfig, GPURequirement, GPUType, CostEstimate, InstanceSpec, InstanceStatus,
)
from agent.cloud.providers.runpod import RunPodProvider
from agent.cloud.providers.vastai import VastAIProvider
from agent.cloud.providers.gpu_soroban import GPUSorobanProvider
from agent.cloud.providers.lambda_labs import LambdaLabsProvider
from agent.cloud.providers.major_clouds import AWSProvider, AzureProvider, GCPProvider


class TestGPURequirement:
    def test_defaults(self):
        gpu = GPURequirement()
        assert gpu.gpu_type == GPUType.A100_80GB
        assert gpu.count == 8
        assert gpu.memory_gb == 80

    def test_custom(self):
        gpu = GPURequirement(gpu_type=GPUType.H100, count=4, min_vram_per_gpu_gb=80)
        assert gpu.gpu_type == GPUType.H100
        assert gpu.count == 4


class TestLaunchConfig:
    def test_defaults(self):
        cfg = LaunchConfig()
        assert cfg.model_scale == "medium"
        assert cfg.use_spot is True
        assert cfg.auto_shutdown_hours == 24.0

    def test_startup_script(self):
        cfg = LaunchConfig(model_scale="small")
        provider = RunPodProvider()
        script = provider.generate_startup_script(cfg)
        assert "tinyllm" in script.lower() or "DS4SmallestAIprjct" in script
        assert "train" in script.lower() or "agent.phase7" in script


class TestCostEstimate:
    def test_summary(self):
        est = CostEstimate(provider="test", gpu_type="a100-80gb", gpu_count=8, price_per_gpu_hour=2.0, price_per_hour_total=16.0, estimated_hours=24, estimated_total_cost=384.0, spot_price_total=192.0)
        assert "384" in est.summary() or "192" in est.summary()
        assert "test" in est.summary()


class TestRunPodProvider:
    def test_init(self):
        p = RunPodProvider(api_key="test-key")
        assert p.NAME == "runpod"
        assert p.api_key == "test-key"

    def test_estimate_cost(self):
        p = RunPodProvider()
        cfg = LaunchConfig(model_scale="medium", auto_shutdown_hours=10)
        est = p.estimate_cost(cfg)
        assert est.provider == "runpod"
        assert est.gpu_count == 8
        assert est.estimated_hours == 10
        assert est.price_per_hour_total > 0

    def test_get_available_gpus(self):
        p = RunPodProvider()
        gpus = p.get_available_gpus()
        assert len(gpus) > 3
        assert any(g["type"] == "a100-80gb" for g in gpus)

    def test_gpu_type_mapping(self):
        p = RunPodProvider()
        assert "A100" in p._gpu_type_id(GPUType.A100_80GB)
        assert "H100" in p._gpu_type_id(GPUType.H100)

    def test_generate_startup_script(self):
        p = RunPodProvider()
        cfg = LaunchConfig(model_scale="xlarge", dataset_path="s3://my-bucket/data")
        script = p.generate_startup_script(cfg)
        assert "aws s3 sync" in script


class TestVastAIProvider:
    def test_init(self):
        p = VastAIProvider(api_key="test")
        assert p.NAME == "vastai"

    def test_estimate_cost(self):
        p = VastAIProvider()
        est = p.estimate_cost(LaunchConfig(model_scale="small", auto_shutdown_hours=5))
        assert est.provider == "vastai"
        assert est.gpu_count == 8


class TestGPUSorobanProvider:
    def test_estimate_cost(self):
        p = GPUSorobanProvider()
        est = p.estimate_cost(LaunchConfig(model_scale="medium"))
        assert est.provider == "gpu_soroban"


class TestAllProviders:
    """Test all providers implement the interface correctly."""

    PROVIDERS = [RunPodProvider, VastAIProvider, GPUSorobanProvider, LambdaLabsProvider, AWSProvider, AzureProvider, GCPProvider]

    def test_all_have_name(self):
        for cls in self.PROVIDERS:
            p = cls()
            assert p.NAME, f"{cls.__name__} has no NAME"
            assert len(p.NAME) > 0

    def test_all_estimate_cost(self):
        cfg = LaunchConfig()
        for cls in self.PROVIDERS:
            p = cls()
            est = p.estimate_cost(cfg)
            assert isinstance(est, CostEstimate), f"{cls.__name__}.estimate_cost() failed"
            assert est.price_per_hour_total > 0

    def test_all_have_gpu_list(self):
        for cls in self.PROVIDERS:
            p = cls()
            gpus = p.get_available_gpus()
            assert isinstance(gpus, list), f"{cls.__name__}.get_available_gpus() failed"
            assert len(gpus) > 0, f"{cls.__name__} returned empty GPU list"

    def test_all_generate_startup_script(self):
        cfg = LaunchConfig(model_scale="nano")
        for cls in self.PROVIDERS:
            p = cls()
            script = p.generate_startup_script(cfg)
            assert len(script) > 0, f"{cls.__name__} generated empty script"

    def test_all_spot_discount_valid(self):
        for cls in self.PROVIDERS:
            p = cls()
            if hasattr(p, 'SPOT_DISCOUNT'):
                assert 0 < p.SPOT_DISCOUNT < 1, f"{cls.__name__} SPOT_DISCOUNT invalid"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
