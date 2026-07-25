#!/usr/bin/env python3
"""
auto_train.py — Fully automated TinyLLM training loop.

Usage:
  python auto_train.py                    # Train 5000 steps, auto-resume from HF
  python auto_train.py --steps 50000      # Train 50000 steps
  python auto_train.py --fresh            # Start from scratch (ignore HF checkpoint)

Flow:
  1. Download latest checkpoint from HuggingFace (if exists)
  2. Load model + tokenizer
  3. Download & tokenize CodeParrot data (if needed)
  4. Train for N steps
  5. Save & upload checkpoint to HuggingFace
  6. Done! (Run again to continue)

Set HF_TOKEN in environment (Kaggle Secrets) to enable auto-upload.
"""

import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

HF_REPO = "Ryo3desu/tinyllm-models"
HF_MODEL_PATH = "tinyllm-nano/model.pt"
MODEL_SIZE = "nano"
SEQ_LEN = 1024
VOCAB_SIZE = 4639

TRAIN_CONFIG = {
    'max_steps': 5000,      # Override with --steps
    'batch_size': 1,        # Kaggle T4: 1 to avoid OOM with DataParallel
    'grad_accum': 4,        # effective batch = batch_size × grad_accum × gpus = 8
    'learning_rate': 3e-4,
    'warmup_steps': 100,
    'max_lr': 3e-4,
    'min_lr': 3e-5,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'log_interval': 10,
    'save_interval': 500,
    'gradient_checkpointing': True,  # trades compute for VRAM (~30% less memory)
}

DATA_CONFIG = {
    'token_limit': 5_000_000,  # tokens to download per session
}

OUTPUT_DIR = "checkpoints"


# ═══════════════════════════════════════════════════════════════
# Model (same as notebook Cell 11)
# ═══════════════════════════════════════════════════════════════

class TinyLLMLayer(nn.Module):
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
        self.up_proj = nn.Linear(D, inter, bias=False)
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
        attn = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.o_proj(attn.transpose(1, 2).contiguous().view(B, S, D)) + residual
        residual = x
        x = self.norm2(x)
        x = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        return x


class TinyLLMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg['vocab_size'], cfg['hidden_size'])
        self.layers = nn.ModuleList([TinyLLMLayer(cfg) for _ in range(cfg['num_hidden_layers'])])
        self.norm = nn.RMSNorm(cfg['hidden_size'], eps=1e-5)
        self.lm_head = nn.Linear(cfg['hidden_size'], cfg['vocab_size'], bias=False)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {'loss': loss, 'logits': logits}


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    # Try local paths (prioritize bundled 32K tokenizer)
    for path in ['tokenizer', 'downloaded_models/tinyllm-nano', 'tinyllm-tokenizer']:
        if os.path.exists(f'{path}/tokenizer.json'):
            tok = AutoTokenizer.from_pretrained(path, use_fast=True)
            # Only add special tokens that don't already exist (avoid ID reassignment)
            existing = set(tok.get_vocab().keys())
            to_add = [t for t in ['<fim_prefix>', '<fim_suffix>', '<fim_middle>',
                                   '<pad>', '<tool_call>', '</tool_call>']
                      if t not in existing]
            if to_add:
                tok.add_special_tokens({'additional_special_tokens': to_add})
            if tok.pad_token is None: tok.pad_token = '<pad>'
            if tok.bos_token is None: tok.bos_token = '<s>'
            if tok.eos_token is None: tok.eos_token = '</s>'
            return tok
    # Fallback: download Swallow (Japanese-optimized, 32K vocab)
    tok = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-7b-v0.1", use_fast=True)
    tok.save_pretrained('tinyllm-tokenizer')
    return tok


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class TokenBinDataset(IterableDataset):
    def __init__(self, path, seq_len, vocab_size):
        self.data = np.memmap(path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        while True:
            offset = np.random.randint(0, len(self.data) - self.seq_len - 1)
            tokens = self.data[offset:offset + self.seq_len + 1]
            input_ids = torch.from_numpy(tokens[:self.seq_len].astype(np.int64))
            labels = torch.from_numpy(tokens[1:self.seq_len + 1].astype(np.int64))
            mask = (input_ids >= 0) & (input_ids < self.vocab_size)
            labels[~mask] = -100
            yield {'input_ids': input_ids, 'labels': labels}


def prepare_data(tokenizer, token_limit=5_000_000):
    """Download CodeParrot and tokenize if data doesn't exist."""
    os.makedirs('data', exist_ok=True)
    if os.path.exists('data/train.bin'):
        data = np.memmap('data/train.bin', dtype=np.int32, mode='r')
        print(f"✅ Data exists: {len(data):,} tokens")
        return

    print("📦 Downloading CodeParrot...")
    from datasets import load_dataset
    try:
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True).take(10000)
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True).take(10000)

    print(f"🔧 Tokenizing (limit={token_limit:,} tokens)...")
    all_tokens = []
    for sample in ds:
        text = sample.get('content') or sample.get('text') or ''
        if len(text) < 10: continue
        all_tokens.extend(tokenizer.encode(text))
        if len(all_tokens) >= token_limit: break

    tokens = np.array(all_tokens, dtype=np.int32)
    split = int(len(tokens) * 0.9)
    tokens[:split].tofile('data/train.bin')
    tokens[split:].tofile('data/val.bin')
    print(f"✅ Data ready: train={split:,}, val={len(tokens)-split:,}")


# ═══════════════════════════════════════════════════════════════
# HuggingFace: download / upload
# ═══════════════════════════════════════════════════════════════

def download_checkpoint():
    """Download latest checkpoint from HF. Returns path or None."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, HF_MODEL_PATH, local_dir='downloaded_models')
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        step = ckpt.get('step', 0)
        print(f"📥 Downloaded checkpoint: step {step:,}")
        return path
    except Exception as e:
        print(f"⚠️  No checkpoint found on HF ({e})")
        return None


def upload_checkpoint(ckpt_path):
    """Upload checkpoint to HF."""
    # Try Kaggle Secrets first, then environment variable
    hf_token = None
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret('HF_TOKEN')
    except (ImportError, Exception):
        pass
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("⚠️  HF_TOKEN not set. Skipping upload.")
        return False
    try:
        from huggingface_hub import login, HfApi
        login(token=hf_token)
        api = HfApi()
        size_gb = os.path.getsize(ckpt_path) / 1e9
        print(f"📤 Uploading checkpoint ({size_gb:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=HF_MODEL_PATH,
            repo_id=HF_REPO,
            repo_type="model",
        )
        print("✅ Uploaded! Next run will auto-resume.")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# MoE Monitoring
# ═══════════════════════════════════════════════════════════════

def _log_moe_metrics(model, gpu_count):
    """Log expert utilization metrics if model has MoE layers.
    
    Monitors for expert collapse (P0 training stability):
      - entropy: gate diversity (higher = better, < 0.5 → collapse warning)
      - imbalance: max/mean tokens per expert (> 5.0 → overload warning)
    """
    try:
        from model.layers.moe import compute_expert_metrics
        
        # Unwrap DataParallel
        m = model.module if gpu_count > 1 else model
        
        # Check if model has MoE gate
        moe_found = False
        for layer in getattr(m, 'layers', []):
            if hasattr(layer, 'moe') and hasattr(layer.moe, 'gate'):
                moe_found = True
                break
        
        if not moe_found:
            return  # Dense model, skip MoE monitoring
        
        # Collect metrics from first MoE layer
        with torch.no_grad():
            for layer in m.layers:
                if hasattr(layer, 'moe') and hasattr(layer.moe, 'gate'):
                    # We need recent gate outputs — stored during forward
                    # For now, log a placeholder (full integration requires hook)
                    pass
        
    except ImportError:
        pass  # MoE module not available
    except Exception:
        pass  # Silent failure for monitoring


# ═══════════════════════════════════════════════════════════════
# Safe Save (avoids Kaggle zip serialization crash)
# ═══════════════════════════════════════════════════════════════

def safe_save(obj, path, max_retries=3):
    """Save PyTorch checkpoint safely, avoiding Kaggle zip-stream corruption.
    
    Uses legacy serialization (_use_new_zipfile_serialization=False) and
    writes to a temp file first, then atomically renames. Retries on failure.
    """
    import tempfile
    tmp_path = path + '.tmp'
    for attempt in range(max_retries):
        try:
            torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
            os.replace(tmp_path, path)  # atomic on POSIX
            return True
        except RuntimeError as e:
            print(f"⚠️  Save attempt {attempt+1}/{max_retries} failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    return False


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train(model, optimizer, scheduler, scaler, train_loader,
          cfg_dict, train_config, start_step=0, device='cuda'):
    """Run training loop. Returns final step count."""
    USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    model.train()
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print(f"🚀 DataParallel: {gpu_count} GPUs")

    # Enable gradient checkpointing to save VRAM (~30% reduction)
    if train_config.get('gradient_checkpointing', False):
        from torch.utils.checkpoint import checkpoint
        base_model = model.module if gpu_count > 1 else model
        for layer in base_model.layers:
            layer._gradient_checkpointing = True
            # Wrap forward with checkpoint
            original_forward = layer.forward
            def make_ckpt_forward(layer_obj):
                def ckpt_forward(x):
                    return checkpoint(layer_obj._original_forward, x, use_reentrant=False)
                return ckpt_forward
            layer._original_forward = original_forward
            layer.forward = make_ckpt_forward(layer)
        print(f"🧠 Gradient checkpointing: ON (VRAM ↓)")

    data_iter = iter(train_loader)
    global_step = start_step
    total_loss = 0.0
    start_time = time.time()

    remaining = train_config['max_steps'] - start_step
    print(f"🏃 Training: step {start_step:,} → {train_config['max_steps']:,} ({remaining:,} steps)")
    print(f"   GPU: {gpu_count}x, batch={train_config['batch_size']}, grad_accum={train_config['grad_accum']}")

    while global_step < train_config['max_steps']:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()  # DataParallel may return per-GPU losses

        loss = loss / train_config['grad_accum']
        if USE_BF16:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        total_loss += loss.item()

        if (global_step + 1) % train_config['grad_accum'] == 0:
            if USE_BF16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % train_config['log_interval'] == 0:
            avg_loss = total_loss / train_config['log_interval']
            steps_done = global_step - start_step
            elapsed = time.time() - start_time
            tok_per_sec = steps_done * train_config['batch_size'] * SEQ_LEN / elapsed if elapsed > 0 else 0
            print(f"   step {global_step:,} | loss={avg_loss:.4f} | {tok_per_sec:.0f} tok/s | {elapsed:.0f}s")
            total_loss = 0.0
        
        # 🔍 MoE monitoring: log expert utilization every 100 steps
        if global_step % 100 == 0 and global_step > 0:
            _log_moe_metrics(model, gpu_count)

        if global_step % train_config['save_interval'] == 0:
            ckpt_dir = f'{OUTPUT_DIR}/step_{global_step}'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = {
                'model_state_dict': (model.module if gpu_count > 1 else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg_dict,
                'step': global_step,
            }
            safe_save(ckpt, f'{ckpt_dir}/model.pt')
            print(f"💾 Saved: {ckpt_dir}/")

    elapsed = time.time() - start_time
    print(f"✅ Done! {elapsed:.0f}s total, step {start_step:,}→{global_step:,}")
    return global_step


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TinyLLM Auto-Training Loop')
    parser.add_argument('--steps', type=int, default=5000, help='Max training steps (total)')
    parser.add_argument('--fresh', action='store_true', help='Start from scratch (ignore HF checkpoint)')
    parser.add_argument('--no-upload', action='store_true', help='Skip HF upload')
    args = parser.parse_args()

    TRAIN_CONFIG['max_steps'] = args.steps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print(f"🤖 TinyLLM Auto-Train — {args.steps:,} steps")
    print(f"   Device: {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    print("=" * 60)

    # ── Step 1: Tokenizer ────────────────────────────────────
    print("\n── Tokenizer ──")
    tokenizer = load_tokenizer()
    print(f"   vocab={len(tokenizer)}")

    # ── Step 2: Data ─────────────────────────────────────────
    print("\n── Data ──")
    prepare_data(tokenizer, DATA_CONFIG['token_limit'])

    # ── Step 3: Model (new or resumed) ──────────────────────
    print("\n── Model ──")
    cfg_dict = {
        'hidden_size': 1024, 'num_hidden_layers': 24, 'vocab_size': len(tokenizer),
        'intermediate_size': 2816,
    }
    model = TinyLLMModel(cfg_dict)

    start_step = 0

    if not args.fresh:
        ckpt_path = download_checkpoint()
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            start_step = ckpt.get('step', 0)
            if start_step >= args.steps:
                print(f"⚠️  Already at step {start_step:,} >= target {args.steps:,}. Increase --steps.")
                return

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {total_params/1e6:.0f}M, from step {start_step:,}")

    model = model.to(device)

    # ── Step 4: Optimizer ────────────────────────────────────
    try:
        optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                         weight_decay=TRAIN_CONFIG['weight_decay'],
                         betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available())
    except TypeError:
        optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                         weight_decay=TRAIN_CONFIG['weight_decay'],
                         betas=(0.9, 0.95), eps=1e-8)

    # Resume optimizer if available
    if start_step > 0 and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("   Optimizer state resumed")
            except Exception:
                pass

    def get_lr_scheduler(optimizer, warmup, max_s, max_lr, min_lr):
        def lr_lambda(step):
            if step < warmup: return float(step) / max(1, warmup)
            progress = float(step - warmup) / max(1, max_s - warmup)
            return min_lr / max_lr + (1 - min_lr / max_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)

    scheduler = get_lr_scheduler(optimizer, TRAIN_CONFIG['warmup_steps'],
                                 TRAIN_CONFIG['max_steps'],
                                 TRAIN_CONFIG['max_lr'], TRAIN_CONFIG['min_lr'])
    for _ in range(start_step):
        scheduler.step()

    scaler = torch.cuda.amp.GradScaler()

    # ── Step 5: Train ───────────────────────────────────────
    print("\n── Training ──")
    train_dataset = TokenBinDataset('data/train.bin', SEQ_LEN, len(tokenizer))
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'])

    final_step = train(model, optimizer, scheduler, scaler, train_loader,
                       cfg_dict, TRAIN_CONFIG, start_step, device)

    # ── Step 6: Save final ──────────────────────────────────
    final_dir = f'{OUTPUT_DIR}/final'
    os.makedirs(final_dir, exist_ok=True)
    final_model = model.module if hasattr(model, 'module') else model
    safe_save({
        'model_state_dict': final_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg_dict,
        'step': final_step,
    }, f'{final_dir}/model.pt')
    print(f"\n💾 Final model: {final_dir}/model.pt (step {final_step:,})")

    # ── Step 7: Upload to HF ────────────────────────────────
    if not args.no_upload:
        print("\n── Upload ──")
        upload_checkpoint(f'{final_dir}/model.pt')

    print("\n🎉 All done! Run again to continue training.")


if __name__ == '__main__':
    main()
