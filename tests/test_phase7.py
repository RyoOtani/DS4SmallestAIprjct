"""Tests for Phase 7: Distributed AI Platform."""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Initialize minimal distributed context for testing
import torch
import torch.distributed as dist

_is_dist_initialized = False


def _ensure_dist():
    global _is_dist_initialized
    if not _is_dist_initialized and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1,
                                    init_method="tcp://localhost:29555")
        except Exception:
            pass  # port in use
        _is_dist_initialized = True


def test_distributed_config():
    """Test distributed config validation."""
    _ensure_dist()
    from agent.phase7.parallelism import DistributedConfig

    # Valid config
    cfg = DistributedConfig(world_size=8, dp_size=4, tp_size=2, pp_size=1)
    cfg.validate()

    # Invalid config
    cfg2 = DistributedConfig(world_size=8, dp_size=3, tp_size=2, pp_size=1)
    try:
        cfg2.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("✓ test_distributed_config passed")


def test_parallel_group_manager():
    """Test process group layout computation."""
    _ensure_dist()
    from agent.phase7.parallelism import ParallelGroupManager, DistributedConfig

    # Use world_size=1 for single-process testing
    cfg = DistributedConfig(world_size=1, dp_size=1, tp_size=1, pp_size=1, rank=0)
    mgr = ParallelGroupManager(cfg)

    assert mgr.dp_rank == 0
    assert mgr.tp_rank == 0
    assert mgr.pp_rank == 0

    print("✓ test_parallel_group_manager passed")


def test_pipeline_schedule():
    """Test 1F1B pipeline schedule generation."""
    from agent.phase7.parallelism import PipelineSchedule

    schedule = PipelineSchedule(pp_size=4, pp_rank=2, n_microbatches=8)
    sched = schedule.get_schedule()

    forwards = [a for a, _ in sched if a == "forward"]
    backwards = [a for a, _ in sched if a == "backward"]

    assert len(forwards) == 8, f"Expected 8 forwards, got {len(forwards)}"
    assert len(backwards) == 8, f"Expected 8 backwards, got {len(backwards)}"

    print("✓ test_pipeline_schedule passed")


def test_mixed_precision_config():
    """Test mixed precision configuration."""
    from agent.phase7.mixed_precision import MixedPrecisionConfig

    cfg = MixedPrecisionConfig(enabled=True, dtype="bfloat16")
    assert cfg.torch_dtype == torch.bfloat16

    cfg16 = MixedPrecisionConfig(enabled=True, dtype="float16")
    assert cfg16.torch_dtype == torch.float16

    cfg32 = MixedPrecisionConfig(enabled=False, dtype="float32")
    assert cfg32.torch_dtype == torch.float32

    print("✓ test_mixed_precision_config passed")


def test_loss_scaler():
    """Test dynamic loss scaling."""
    from agent.phase7.mixed_precision import DynamicLossScaler, MixedPrecisionConfig

    cfg = MixedPrecisionConfig(dtype="float16", loss_scale=65536.0,
                                growth_interval=5, hysteresis=1)
    scaler = DynamicLossScaler(cfg)

    # Initial scale
    assert scaler.scale == 65536.0

    # Simulate overflow
    scaler.update(overflow=True)
    assert scaler.scale == 32768.0  # halved

    # Many non-overflow steps should grow scale
    for _ in range(cfg.growth_interval + 100):
        scaler.update(overflow=False)

    assert scaler.scale >= 65536.0  # grew

    print("✓ test_loss_scaler passed")


def test_column_parallel_linear():
    """Test tensor parallel column linear layer (single-process)."""
    import torch
    from agent.phase7.parallelism import ColumnParallelLinear

    # tp_size=1, tp_rank=0 → behaves like regular Linear
    layer = ColumnParallelLinear(512, 1024, tp_size=1, tp_rank=0, gather_output=True)

    x = torch.randn(2, 32, 512)
    y = layer(x)

    assert y.shape == (2, 32, 1024), f"Expected (2, 32, 1024), got {y.shape}"
    print(f"    Output shape: {y.shape}")

    print("✓ test_column_parallel_linear passed")


if __name__ == "__main__":
    print("════ Phase 7 Tests ════")
    test_distributed_config()
    test_parallel_group_manager()
    test_pipeline_schedule()
    test_mixed_precision_config()
    test_loss_scaler()
    test_column_parallel_linear()
    print("\n✅ All Phase 7 tests passed!")
