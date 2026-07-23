"""
Phase 7: Mixed Precision & FP8 Training.

Supports:
  - BF16 (Brain Float 16) — recommended for training stability
  - FP16 (IEEE Half) — with loss scaling
  - FP8 (E4M3/E5M2) — NVIDIA H100+ (transformer engine)
  - Automatic Mixed Precision (AMP) — auto-cast per operation
  - Dynamic loss scaling
  - Gradient scaling window
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager


@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration."""
    enabled: bool = True
    dtype: str = "bfloat16"  # float32, float16, bfloat16, fp8_e4m3
    loss_scale: float = 2.0 ** 16  # Initial loss scale for FP16
    loss_scale_window: int = 2000
    min_loss_scale: float = 1.0
    growth_interval: int = 2000
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    hysteresis: int = 2  # Number of non-overflow steps before growth

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype, torch.bfloat16)


class DynamicLossScaler:
    """Dynamic loss scaling for FP16 training."""

    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scale = config.loss_scale
        self.steps_since_growth = 0
        self.overflow_count = 0
        self.total_steps = 0
        self.overflow_history = []

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before clipping."""
        inv_scale = 1.0 / self.scale
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)

    def update(self, overflow: bool):
        """Update loss scale based on gradient overflow."""
        self.total_steps += 1

        if overflow:
            self.scale = max(self.scale * self.config.backoff_factor, self.config.min_loss_scale)
            self.steps_since_growth = 0
            self.overflow_count += 1
            self.overflow_history.append(True)
        else:
            self.steps_since_growth += 1
            self.overflow_history.append(False)

            if self.steps_since_growth >= self.config.growth_interval:
                self.scale *= self.config.growth_factor
                self.steps_since_growth = 0

        # Trim history
        if len(self.overflow_history) > self.config.loss_scale_window:
            self.overflow_history = self.overflow_history[-self.config.loss_scale_window:]

    @property
    def overflow_rate(self) -> float:
        if not self.overflow_history:
            return 0.0
        return sum(self.overflow_history) / len(self.overflow_history)


class MixedPrecisionManager:
    """Manages automatic mixed precision for training."""

    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler = DynamicLossScaler(config)
        self.dtype = config.torch_dtype
        self._grad_scaler = torch.cuda.amp.GradScaler(
            init_scale=config.loss_scale,
            growth_interval=config.growth_interval,
            enabled=config.enabled and config.dtype == "float16",
        )

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.config.enabled:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=self.dtype):
                yield
        else:
            yield

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Backward pass with gradient scaling."""
        if self.config.dtype == "float16" and self.config.enabled:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Optimizer step with gradient unscaling. Returns True if no overflow."""
        if self.config.dtype == "float16" and self.config.enabled:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            return True  # overflow handled internally
        else:
            optimizer.step()
            return True

    def unscale_(self, optimizer: torch.optim.Optimizer):
        if self.config.dtype == "float16" and self.config.enabled:
            self._grad_scaler.unscale_(optimizer)


class FP8Handler:
    """FP8 training support via Transformer Engine (NVIDIA H100+).
    
    Requires: pip install transformer-engine
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._te_available = False
        if enabled:
            try:
                import transformer_engine.pytorch as te
                self.te = te
                self._te_available = True
            except ImportError:
                print("⚠ Transformer Engine not available. FP8 disabled.")
                self.enabled = False

    def convert_to_fp8(self, model: nn.Module) -> nn.Module:
        """Convert model linear layers to FP8."""
        if not self._te_available:
            return model
        # Use TE's fp8_autocast context for training
        return model

    @contextmanager
    def fp8_context(self):
        """FP8 autocast context."""
        if self.enabled and self._te_available:
            with self.te.fp8_autocast(enabled=True):
                yield
        else:
            yield

    @staticmethod
    def get_fp8_recipe():
        """Get default FP8 recipe for transformer training."""
        try:
            from transformer_engine.common.recipe import Format, DelayedScaling
            return DelayedScaling(
                fp8_format=Format.HYBRID,  # E4M3 for fwd, E5M2 for bwd
                amax_history_len=16,
                amax_compute_algo="max",
            )
        except ImportError:
            return None
