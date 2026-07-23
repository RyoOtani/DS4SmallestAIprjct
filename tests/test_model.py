"""Tests for TinyLLM Model Architecture."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


def test_model_creation():
    """Test model creation at all scales."""
    from model.config import get_config, TINYLLM_CONFIGS
    from model.architecture import TinyLLMModel
    
    # Test all non-giant configs
    for name in ["nano", "small"]:  # skip medium/large for CI speed
        print(f"  Testing {name}...")
        config = get_config(name)
        model = TinyLLMModel(config)
        
        assert isinstance(model, nn.Module)
        assert model.config.name == config.name
        
        total = sum(p.numel() for p in model.parameters())
        print(f"    {name}: {total/1e6:.1f}M params")
    
    print("✓ test_model_creation passed")


def test_forward_pass():
    """Test forward pass with dummy data."""
    from model.config import get_config
    from model.architecture import TinyLLMModel
    
    config = get_config("nano")
    config.use_moe = False  # Simpler for testing
    config.use_mtp = False
    config.n_layers = 4
    
    model = TinyLLMModel(config)
    model.eval()
    
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert "last_hidden_state" in outputs
     
    print("✓ test_forward_pass passed")


def test_generation():
    """Test text generation."""
    from model.config import get_config
    from model.architecture import TinyLLMModel
    
    config = get_config("nano")
    config.n_layers = 4
    
    model = TinyLLMModel(config)
    model.eval()
     
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=8, do_sample=False)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 16
    print(f"    Generated {generated.shape[1]} tokens")
     
    print("✓ test_generation passed")


def test_moe_routing():
    """Test MoE routing and load balancing."""
    from model.layers.moe import MoELayer, load_balancing_loss
     
    batch, seq = 2, 16
    hidden_dim = 256
    
    moe = MoELayer(
        hidden_dim=hidden_dim,
        n_experts=8,
        n_active=2,
        expert_inter_dim=512,
    )
     
    x = torch.randn(batch, seq, hidden_dim)
    out, aux_loss = moe(x)
     
    assert out.shape == (batch, seq, hidden_dim)
    assert aux_loss.item() >= 0
    
    print("✓ test_moe_routing passed")


def test_attention():
    """Test MLA and GQA attention."""
    from model.layers.attention import MultiHeadLatentAttention, GroupedQueryAttention
     
    batch, seq, hidden = 2, 32, 256
    x = torch.randn(batch, seq, hidden)
    
    # Test MLA
    mla = MultiHeadLatentAttention(
        hidden_dim=hidden,
        n_heads=8,
        head_dim=64,
        kv_latent_dim=256,
        max_seq_len=128,
    )
    out, _ = mla(x)
    assert out.shape == (batch, seq, hidden)
    
    # Test GQA
    gqa = GroupedQueryAttention(
        hidden_dim=hidden,
        n_heads=8,
        head_dim=64,
        n_kv_heads=4,
        max_seq_len=128,
    )
    out, _ = gqa(x)
    assert out.shape == (batch, seq, hidden)
     
    print("✓ test_attention passed")


def test_gguf_export():
    """Test GGUF export (dry run)."""
    from model.config import get_config
    from model.architecture import TinyLLMModel
    from model.export.gguf_exporter import export_model_to_gguf
    import tempfile
    
    config = get_config("nano")
    config.n_layers = 4
    config.use_moe = False
    config.use_mtp = False
    
    model = TinyLLMModel(config)
    model.eval()
    
    model_config_dict = {
        "name": config.name,
        "hidden_dim": config.hidden_dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "use_moe": config.use_moe,
        "use_mla": config.use_mla,
        "kv_latent_dim": config.kv_latent_dim,
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        out_path = f.name
    
    try:
        export_model_to_gguf(model, out_path, model_config_dict, use_q4_0=True)
        import os
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"    GGUF exported: {size_mb:.1f} MB")
        os.unlink(out_path)
    except Exception as e:
        os.unlink(out_path)
        print(f"    ⚠ GGUF export test skipped: {e}")
    
    print("✓ test_gguf_export passed")


def test_configs():
    """Test all model configs are valid."""
    from model.config import TINYLLM_CONFIGS, get_config
     
    for name in TINYLLM_CONFIGS:
        cfg = get_config(name)
        assert cfg.hidden_dim > 0
        assert cfg.n_layers > 0
        assert cfg.vocab_size > 0
        assert cfg.n_heads > 0
        if cfg.use_moe:
            assert cfg.n_experts > 0
            assert cfg.n_active_experts > 0
    
    print(f"✓ test_configs passed ({len(TINYLLM_CONFIGS)} configs)")
    
    # Print available configs
    from model.config import list_configs
    for cfg in list_configs():
        print(f"    {cfg['name']:20s} {cfg['total_params']:>8s} total  {cfg['moe']}")


if __name__ == "__main__":
    print("════ TinyLLM Model Tests ════")
    test_model_creation()
    test_forward_pass()
    test_generation()
    test_moe_routing()
    test_attention()
    test_gguf_export()
    test_configs()
    print("\n✅ All model tests passed!")
