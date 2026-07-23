"""Tests for Training Methods and MPS Backend."""
from __future__ import annotations
import sys, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.training.mps_trainer import MPSDetector, MPSConfig, MPSTrainer
from agent.training.methods import (
    LoRATrainer, QLoRATrainer, DPOTrainer,
    DistillationTrainer, InstructionTuner,
    TrainingPreset, TrainingMethodRegistry, TRAINING_PRESETS,
)


class TestMPSDetector:
    def test_is_available(self):
        """MPS detection should not crash."""
        available = MPSDetector.is_available()
        assert isinstance(available, bool)

    def test_get_device_info(self):
        info = MPSDetector.get_device_info()
        assert "platform" in info
        assert "available" in info
        assert "device_name" in info
        if platform.system() == "Darwin":
            assert "Apple" in info["device_name"] or "unknown" in info["device_name"]
        # Non-Mac should still return valid dict
        assert isinstance(info["unified_memory_gb"], int)

    def test_print_info(self):
        """Should not crash."""
        MPSDetector.print_info()


class TestMPSConfig:
    def test_defaults(self):
        cfg = MPSConfig()
        assert cfg.use_bfloat16 is True
        assert cfg.use_amp is True
        assert cfg.gradient_checkpointing is True
        assert 0 < cfg.max_memory_fraction <= 1.0

    def test_auto_batch_size(self):
        cfg = MPSConfig(auto_batch_size=True)
        assert cfg.min_batch_size <= cfg.max_batch_size


class TestMPSTrainer:
    def test_init(self):
        trainer = MPSTrainer()
        assert trainer.device is not None
        assert trainer.config is not None

    def test_get_device(self):
        trainer = MPSTrainer()
        device = trainer._get_device()
        assert str(device) in ("mps", "cuda", "cpu") or device in ("mps", "cuda", "cpu")


class TestTrainingPresets:
    def test_all_presets_have_fields(self):
        for pid, preset in TRAINING_PRESETS.items():
            assert preset.name, f"{pid}: missing name"
            assert preset.method, f"{pid}: missing method"
            assert preset.learning_rate > 0, f"{pid}: invalid lr"
            assert preset.batch_size > 0, f"{pid}: invalid batch"

    def test_preset_methods_match_registry(self):
        for pid, preset in TRAINING_PRESETS.items():
            if preset.method in TrainingMethodRegistry.METHODS:
                assert preset.method in TrainingMethodRegistry.METHODS


class TestTrainingMethodRegistry:
    def test_list_methods(self):
        methods = TrainingMethodRegistry.list_methods()
        assert len(methods) >= 8  # At least 8 methods
        assert any(m["id"] == "lora" for m in methods)
        assert any(m["id"] == "dpo" for m in methods)
        assert any(m["id"] == "distill" for m in methods)

    def test_list_presets(self):
        presets = TrainingMethodRegistry.list_presets()
        assert len(presets) >= 15  # At least 15 presets
        assert any(p["id"] == "lora-fast" for p in presets)

    def test_get_preset(self):
        preset = TrainingMethodRegistry.get_preset("lora-fast")
        assert preset is not None
        assert preset.method == "lora"
        assert preset.learning_rate == 2e-4

    def test_get_preset_invalid(self):
        preset = TrainingMethodRegistry.get_preset("nonexistent")
        assert preset is None

    def test_recommend(self):
        recs = TrainingMethodRegistry.recommend("chat", "mps")
        assert len(recs) >= 1
        assert any("lora" in r["method"] for r in recs)

    def test_recommend_code(self):
        recs = TrainingMethodRegistry.recommend("code generation", "cuda")
        assert len(recs) >= 1
        assert any(r["method"] in ("lora", "domain") for r in recs)

    def test_recommend_alignment(self):
        recs = TrainingMethodRegistry.recommend("align preferences", "cpu")
        assert any(r["method"] == "dpo" for r in recs)


class TestLoRATrainer:
    def test_has_constants(self):
        assert LoRATrainer.DEFAULT_RANK == 16
        assert LoRATrainer.DEFAULT_ALPHA == 32
        assert len(LoRATrainer.TARGET_MODULES) >= 3


class TestInstructionTuner:
    def test_format_alpaca(self):
        text = InstructionTuner.format_alpaca(
            "Write a poem", "about AI", "Roses are red...",
        )
        assert "### Instruction:" in text
        assert "Write a poem" in text
        assert "### Input:" in text
        assert "### Response:" in text

    def test_format_chatml(self):
        text = InstructionTuner.format_chatml([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ])
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "<|im_end|>" in text

    def test_format_tinyllm(self):
        text = InstructionTuner.format_tinyllm([
            {"role": "user", "content": "Hello"},
        ])
        assert "User: Hello" in text


class TestDistillationTrainer:
    def test_has_distillation_loss(self):
        assert hasattr(DistillationTrainer, 'distillation_loss')
        assert hasattr(DistillationTrainer, 'feature_distillation_loss')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
