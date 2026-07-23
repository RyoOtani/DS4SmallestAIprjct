#!/usr/bin/env python3
"""Generate config.json for all TinyLLM scales for Hugging Face Hub."""
import json
import os
import sys

# Read config.py source and execute it directly (avoids model/__init__.py → torch chain)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "config.py")
with open(config_path) as f:
    config_source = f.read()

# Execute config.py in an isolated namespace
config_ns = {}
exec(compile(config_source, config_path, "exec"), config_ns)
get_config = config_ns["get_config"]

scales = ['nano', 'small', 'medium', 'large', 'dense-7b', 'xlarge', 'xxlarge', 'mega', 'giga']
os.makedirs('hf_models', exist_ok=True)

for scale in scales:
    cfg = get_config(scale)
    d = {
        'model_type': 'tinyllm',
        'architectures': ['TinyLLMModel'],
        'hidden_size': cfg.hidden_dim,
        'num_hidden_layers': cfg.n_layers,
        'num_attention_heads': cfg.n_heads,
        'num_key_value_heads': cfg.n_kv_heads,
        'head_dim': cfg.head_dim,
        'intermediate_size': cfg.ffn_inter_dim,
        'vocab_size': cfg.vocab_size,
        'max_position_embeddings': cfg.max_seq_len,
        'use_moe': cfg.use_moe,
        'num_experts': cfg.n_experts,
        'num_active_experts': cfg.n_active_experts,
        'expert_intermediate_size': cfg.expert_inter_dim,
        'shared_experts': cfg.shared_experts,
        'use_mla': cfg.use_mla,
        'kv_latent_dim': cfg.kv_latent_dim,
        'use_mtp': cfg.use_mtp,
        'mtp_depth': cfg.mtp_depth,
        'norm_type': cfg.norm_type,
        'norm_eps': cfg.norm_eps,
        'activation_function': cfg.ffn_activation,
        'rope_theta': cfg.rope_theta,
        'sliding_window': cfg.sliding_window,
        'tie_word_embeddings': cfg.tie_word_embeddings,
        'dropout': cfg.dropout,
        'torch_dtype': 'bfloat16',
        'transformers_version': '4.46.0',
    }
    path = f'hf_models/tinyllm-{scale}/config.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    total_m = int(cfg.total_params_estimate // 1_000_000)
    print(f'  ✅ {scale:12s} config.json ({total_m:>6}M total params)')

print(f'\n全{len(scales)}モデル hf_models/ に生成完了')
