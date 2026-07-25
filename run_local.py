#!/usr/bin/env python3
"""
TinyLLM ローカル推論スクリプト
使い方: python run_local.py

1. HF から学習済みモデルを自動ダウンロード
2. 対話モードでテキスト生成
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 設定 ────────────────────────────────────────────────────────
MODEL_DIR = "downloaded_models/tinyllm-nano"
HF_REPO = "Ryo3desu/tinyllm-models"
HF_SUBDIR = "tinyllm-nano"

# 生成パラメータ
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.1


# ═══════════════════════════════════════════════════════════════
# モデル定義 (ノートブック Cell 11 と同じシンプル版)
# ═══════════════════════════════════════════════════════════════

class TinyLLMLayer(nn.Module):
    """Single transformer block: RMSNorm → Attention → RMSNorm → FFN."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg['hidden_size']
        inter = cfg.get('intermediate_size', D * 11 // 4)
        self.norm1 = nn.RMSNorm(D, eps=1e-5)
        self.norm2 = nn.RMSNorm(D, eps=1e-5)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.gate_proj = nn.Linear(D, inter, bias=False)
        self.up_proj   = nn.Linear(D, inter, bias=False)
        self.down_proj = nn.Linear(inter, D, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        B, S, D = x.shape
        n_heads = 16
        head_dim = D // n_heads
        q = self.q_proj(x).view(B, S, n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, n_heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.o_proj(attn.transpose(1, 2).contiguous().view(B, S, D)) + residual
        residual = x
        x = self.norm2(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        return x


class TinyLLMModel(nn.Module):
    """TinyLLM — シンプル transformer for pretraining demo."""
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg['vocab_size'], cfg['hidden_size'])
        self.layers = nn.ModuleList([
            TinyLLMLayer(cfg) for _ in range(cfg['num_hidden_layers'])
        ])
        self.norm = nn.RMSNorm(cfg['hidden_size'], eps=1e-5)
        self.lm_head = nn.Linear(cfg['hidden_size'], cfg['vocab_size'], bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ═══════════════════════════════════════════════════════════════
# トークナイザー読み込み
# ═══════════════════════════════════════════════════════════════

def load_tokenizer(model_dir):
    """Load tokenizer from local dir or bundled one."""
    from transformers import AutoTokenizer

    # Try the downloaded tokenizer first
    if os.path.exists(f"{model_dir}/tokenizer.json"):
        print(f"📥 Tokenizer: {model_dir}/")
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Fallback: bundled tokenizer
    if os.path.exists("tokenizer/tokenizer.json"):
        print("📥 Tokenizer: tokenizer/ (bundled)")
        tok = AutoTokenizer.from_pretrained("tokenizer", use_fast=True)
        tok.add_special_tokens({
            'additional_special_tokens': [
                '<fim_prefix>', '<fim_suffix>', '<fim_middle>',
                '<pad>', '<tool_call>', '</tool_call>',
                '<scratchpad>', '</scratchpad>',
            ]
        })
        if tok.pad_token is None: tok.pad_token = '<pad>'
        if tok.bos_token is None: tok.bos_token = '<s>'
        if tok.eos_token is None: tok.eos_token = '</s>'
        return tok

    raise FileNotFoundError("Tokenizer not found! Run: python create_tokenizer.py")


# ═══════════════════════════════════════════════════════════════
# モデルダウンロード & ロード
# ═══════════════════════════════════════════════════════════════

def download_model(model_dir):
    """Download model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    os.makedirs(model_dir, exist_ok=True)

    files = ['config.json', 'model.pt', 'tokenizer.json', 'tokenizer_config.json']
    for f in files:
        local_path = os.path.join(model_dir, f)
        if os.path.exists(local_path):
            print(f"⏭️  Skip: {f} (already exists)")
            continue
        print(f"📥 Downloading: {f}...")
        hf_hub_download(HF_REPO, f"{HF_SUBDIR}/{f}", local_dir="downloaded_models")
        print(f"   ✅ {f}")

    print("✅ All files ready!")


def load_model(model_dir):
    """Load TinyLLM model from checkpoint."""
    # Load config
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    cfg['vocab_size'] = cfg.get('vocab_size', 65536)

    print(f"📋 Config: hidden={cfg['hidden_size']}, layers={cfg['num_hidden_layers']}, vocab={cfg['vocab_size']}")

    # Create model
    model = TinyLLMModel(cfg)

    # Load weights
    ckpt_path = f"{model_dir}/model.pt"
    print(f"📦 Loading weights: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded! ~{total_params/1e9:.2f}B params, {cfg['num_hidden_layers']} layers")

    return model, cfg


# ═══════════════════════════════════════════════════════════════
# テキスト生成
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS,
             temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
             repetition_penalty=REPETITION_PENALTY):
    """Autoregressive text generation."""
    device = next(model.parameters()).device
    model.eval()

    # Encode prompt
    if tokenizer.bos_token_id is not None:
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
    else:
        input_ids = tokenizer.encode(prompt)

    if len(input_ids) == 0:
        input_ids = [tokenizer.bos_token_id or 0]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = input_ids.copy()
    eos_id = tokenizer.eos_token_id

    start_time = time.time()
    tokens_generated = 0

    for _ in range(max_tokens):
        # Trim to avoid OOM (keep last 2048 tokens)
        if input_tensor.size(1) > 2048:
            input_tensor = input_tensor[:, -2048:]

        # Forward
        logits = model(input_tensor)
        next_logits = logits[0, -1, :] / temperature

        # Repetition penalty
        if repetition_penalty != 1.0:
            for tid in set(generated[-50:]):
                if next_logits[tid] < 0:
                    next_logits[tid] *= repetition_penalty
                else:
                    next_logits[tid] /= repetition_penalty

        # Top-k filtering
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            mask = torch.full_like(next_logits, float('-inf'))
            mask[topk_idx] = topk_vals
            next_logits = mask

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_idx_to_remove = cum_probs > top_p
            sorted_idx_to_remove[0] = False  # keep at least one
            indices_to_remove = sorted_idx[sorted_idx_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        tokens_generated += 1

        # Stop on EOS
        if eos_id is not None and next_token == eos_id:
            break

        # Append to input for next step
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    elapsed = time.time() - start_time
    tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    # Decode
    result = tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)

    print(f"\n⚡ {tokens_generated} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
    return result


# ═══════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🤖 TinyLLM — ローカル推論")
    print("=" * 60)

    # Step 1: Download model if needed
    model_pt = f"{MODEL_DIR}/model.pt"
    if not os.path.exists(model_pt):
        print("\n⚠️  モデルが見つかりません。HFからダウンロードします (1.27GB)...")
        download_model(MODEL_DIR)
    else:
        print(f"\n✅ モデル発見: {model_pt}")

    # Step 2: Load tokenizer
    print("\n── トークナイザー読み込み ──")
    tokenizer = load_tokenizer(MODEL_DIR)
    print(f"   vocab={len(tokenizer)}, bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")

    # Step 3: Load model
    print("\n── モデル読み込み ──")
    model, cfg = load_model(MODEL_DIR)

    # Step 4: Move to device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    model = model.to(device)
    model.eval()

    # Step 5: Interactive loop
    print("\n" + "=" * 60)
    print("💬 対話モード (Ctrl+C で終了、/quit でも終了)")
    print(f"   temp={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 終了します。")
            break

        if not prompt:
            continue
        if prompt.lower() in ('/quit', '/exit', 'exit', 'quit'):
            print("👋 終了します。")
            break

        print("\n🤖 TinyLLM: ", end="", flush=True)
        response = generate(model, tokenizer, prompt)
        print(response)


if __name__ == "__main__":
    main()
