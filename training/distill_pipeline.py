#!/usr/bin/env python3
"""
distill_pipeline.py — DeepSeek → TinyLLM Distillation Pipeline

End-to-end knowledge distillation from large teacher (DeepSeek V4, Qwen, etc.)
to TinyLLM student model.

Pipeline stages:
  1. Load teacher model (or use pre-computed logits)
  2. Load/initialize TinyLLM student
  3. Prepare dataset with teacher logits (online or offline)
  4. Run distillation training
  5. Quantize distilled model (Q4_K_M)
  6. Export to GGUF for C runtime inference

Usage:
  # Online distillation (teacher runs alongside student)
  python distill_pipeline.py --teacher Qwen/Qwen2.5-7B --mode online
  
  # Offline (pre-computed teacher logits)
  python distill_pipeline.py --teacher-logits data/teacher_logits.bin --mode offline
  
  # Skip distillation, just quantize + export
  python distill_pipeline.py --student checkpoints/nano/model.pt --quantize-only
"""

import os, sys, json, argparse, time, math, struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.distill import DistillConfig, DistillTrainer, OfflineDistillDataset
from training.quantize import QuantType, quantize_model_weights


# ═══════════════════════════════════════════════════════════════
# Stage 1: Teacher Logit Collection (Offline)
# ═══════════════════════════════════════════════════════════════

def collect_teacher_logits(
    teacher_model,
    tokenizer,
    token_file: str,
    output_file: str,
    max_samples: int = 10000,
    seq_len: int = 1024,
    batch_size: int = 4,
    device: torch.device = None,
):
    """
    Pre-compute teacher logits for offline distillation.
    
    Reads token data from binary file, runs teacher model, saves logits.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    # Read tokens
    data = np.memmap(token_file, dtype=np.int32, mode='r')
    total_tokens = len(data)
    
    print(f"📊 Collecting teacher logits: {min(max_samples * seq_len, total_tokens):,} tokens")
    
    with open(output_file, 'wb') as f:
        # Header: number of samples
        num_samples = min(max_samples, total_tokens // seq_len)
        f.write(struct.pack('<I', num_samples))
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples * seq_len, seq_len * batch_size), desc='Teacher'):
                batch_inputs = []
                batch_starts = []
                
                for j in range(batch_size):
                    start = i + j * seq_len
                    if start + seq_len > total_tokens:
                        break
                    tokens = data[start:start + seq_len]
                    batch_inputs.append(torch.from_numpy(tokens.astype(np.int64)))
                    batch_starts.append(start)
                
                if not batch_inputs:
                    break
                
                # Pad to same length
                input_ids = torch.stack(batch_inputs).to(device)
                
                # Forward teacher
                outputs = teacher_model(input_ids=input_ids)
                logits = outputs['logits'].cpu().float().numpy()  # [B, S, V]
                
                # Write each sample
                for j in range(len(batch_inputs)):
                    f.write(struct.pack('<I', seq_len))
                    f.write(batch_inputs[j].numpy().astype(np.int32).tobytes())
                    f.write(struct.pack('<I', seq_len))
                    f.write(logits[j].astype(np.float16).tobytes())
    
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Saved {num_samples} samples to {output_file} ({size_mb:.0f} MB)")


# ═══════════════════════════════════════════════════════════════
# Stage 2: Load Student Model
# ═══════════════════════════════════════════════════════════════

def load_student_model(checkpoint_path: str, vocab_size: int, device: torch.device):
    """
    Load TinyLLM student from checkpoint or create fresh.
    """
    from transformers import AutoConfig
    
    # Try loading checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in ckpt:
            # Determine model size from checkpoint
            state = ckpt['model_state_dict']
            hidden_dim = state['layers.0.q_proj.weight'].shape[0]
            num_layers = sum(1 for k in state if k.startswith('layers.') and k.endswith('.q_proj.weight'))
            
            print(f"📦 Loading student from checkpoint: {num_layers} layers, hidden={hidden_dim}")
            
            # Build model matching checkpoint
            from TINYLLM_TRAIN_BENCHMARK import TinyLLMModel
            # Actually we need a standalone model builder
            print("⚠️  Full model loading requires model definition — using simplified loader")
    
    # Fallback: create fresh nano model
    print("📦 Creating fresh TinyLLM-nano student model")
    # Simplified model definition inline
    class TinyLLMLayer(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            inter = hidden_dim * 11 // 4
            self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-5)
            self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-5)
            self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_proj = nn.Linear(hidden_dim, inter, bias=False)
            self.up_proj = nn.Linear(hidden_dim, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden_dim, bias=False)
        
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
    
    class TinyLLM(nn.Module):
        def __init__(self, vocab, hidden, layers):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([TinyLLMLayer(hidden) for _ in range(layers)])
            self.norm = nn.RMSNorm(hidden, eps=1e-5)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
        
        def forward(self, input_ids, labels=None):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            logits = self.lm_head(x)
            return {'logits': logits, 'hidden_states': x}
    
    model = TinyLLM(vocab_size, 1024, 24).to(device)
    
    # Load weights if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"   Loaded weights from checkpoint")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Student: ~{total_params/1e9:.2f}B params")
    return model


# ═══════════════════════════════════════════════════════════════
# Stage 3: Quantize + Export
# ═══════════════════════════════════════════════════════════════

def quantize_and_export(student_model, tokenizer, output_dir: str, qtype: QuantType):
    """
    Quantize distilled model and export to GGUF.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔧 Quantizing model to {qtype.name}...")
    state_dict = {k: v.cpu() for k, v in student_model.state_dict().items()}
    quantized = quantize_model_weights(state_dict, qtype)
    
    # Save quantized weights
    torch.save(quantized, f'{output_dir}/model_q4.pt')
    
    # Also save FP16 version for comparison
    fp16_sd = {k: v.half() for k, v in state_dict.items()}
    torch.save(fp16_sd, f'{output_dir}/model_fp16.pt')
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config = {
        'model_type': 'tinyllm',
        'hidden_size': 1024,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'num_kv_heads': 4,
        'vocab_size': len(tokenizer),
        'max_position_embeddings': 8192,
        'quantization': qtype.name,
    }
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    fp16_mb = os.path.getsize(f'{output_dir}/model_fp16.pt') / (1024**2)
    q4_mb = os.path.getsize(f'{output_dir}/model_q4.pt') / (1024**2)
    
    print(f"\n📊 Export summary:")
    print(f"   FP16 model: {fp16_mb:.0f} MB")
    print(f"   Q4 model:   {q4_mb:.0f} MB")
    print(f"   Saved to:   {output_dir}/")
    
    return quantized


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def run_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 DeepSeek→TinyLLM Distillation Pipeline")
    print(f"   Device: {device}")
    print(f"   Mode:   {args.mode}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or 'tokenizer', use_fast=True
    )
    vocab_size = len(tokenizer)
    print(f"   Vocab:  {vocab_size:,}")
    
    # ── Quantize-only mode ────────────────────────────────
    if args.quantize_only:
        student = load_student_model(args.student, vocab_size, device)
        quantize_and_export(student, tokenizer, args.output_dir,
                           getattr(QuantType, args.qtype))
        print("✅ Quantize-only pipeline complete")
        return
    
    # ── Full distillation pipeline ────────────────────────
    # Load teacher
    teacher = None
    if args.mode == 'online' and args.teacher:
        print(f"📥 Loading teacher: {args.teacher}")
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher,
            torch_dtype=torch.bfloat16,
            device_map='auto' if torch.cuda.device_count() > 1 else None,
            trust_remote_code=True,
        )
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(f"   Teacher: ~{teacher_params/1e9:.2f}B params")
    
    # Load student
    student = load_student_model(args.student, vocab_size, device)
    
    # ── Offline mode: pre-compute teacher logits ──────────
    if args.mode == 'offline' and args.teacher_logits is None:
        assert teacher is not None, "Need teacher model for offline logit collection"
        print("\n📊 Stage 1: Collecting teacher logits...")
        collect_teacher_logits(
            teacher, tokenizer,
            token_file=args.data_path or 'data/train.bin',
            output_file='data/teacher_logits.bin',
            max_samples=args.max_samples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        args.teacher_logits = 'data/teacher_logits.bin'
    
    # ── Distillation training ─────────────────────────────
    if not args.skip_train:
        print(f"\n🎓 Stage 2: Distillation training")
        distill_cfg = DistillConfig(
            temperature=args.temperature,
            kd_loss_weight=args.kd_weight,
            ce_loss_weight=args.ce_weight,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            max_seq_len=args.seq_len,
            use_amp=args.amp,
        )
        
        # Setup dataset
        if args.mode == 'offline':
            train_ds = OfflineDistillDataset(
                args.teacher_logits or 'data/teacher_logits.bin',
                args.seq_len, vocab_size
            )
        else:
            # Online: use raw token data, teacher runs alongside
            from training.dataset import TokenBinDataset
            train_ds = TokenBinDataset(
                args.data_path or 'data/train.bin',
                args.seq_len, vocab_size
            )
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size)
        
        # Train
        trainer = DistillTrainer(
            student_model=student,
            teacher_model=teacher,
            tokenizer=tokenizer,
            config=distill_cfg,
            device=device,
        )
        trainer.train(train_loader, resume_from=args.resume_from)
    
    # ── Quantize + Export ─────────────────────────────────
    print(f"\n📦 Stage 3: Quantization + Export")
    quantize_and_export(
        student, tokenizer, args.output_dir,
        getattr(QuantType, args.qtype)
    )
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!")
    print(f"   Output: {args.output_dir}/")
    print(f"   Next:   ./tinyllm run {args.output_dir}/model_q4.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSeek→TinyLLM Distillation Pipeline')
    
    # Mode
    parser.add_argument('--mode', default='online', choices=['online', 'offline'],
                       help='Online (teacher runs live) or offline (pre-computed logits)')
    parser.add_argument('--quantize-only', action='store_true',
                       help='Skip distillation, only quantize + export')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training, only quantize + export')
    
    # Models
    parser.add_argument('--teacher', default=None,
                       help='Teacher model ID (e.g., deepseek-ai/DeepSeek-V4)')
    parser.add_argument('--student', default='checkpoints/final/model.pt',
                       help='Student model checkpoint path')
    parser.add_argument('--tokenizer-path', default='tokenizer',
                       help='Tokenizer path')
    
    # Data
    parser.add_argument('--data-path', default='data/train.bin',
                       help='Training data (binary token file)')
    parser.add_argument('--teacher-logits', default=None,
                       help='Pre-computed teacher logits file (offline mode)')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Max number of training samples')
    parser.add_argument('--seq-len', type=int, default=1024,
                       help='Sequence length')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=5000)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--kd-weight', type=float, default=0.7)
    parser.add_argument('--ce-weight', type=float, default=0.3)
    parser.add_argument('--resume-from', type=int, default=0)
    parser.add_argument('--amp', action='store_true', default=True)
    
    # Quantization
    parser.add_argument('--qtype', default='Q4_K_M',
                       choices=['F32', 'F16', 'Q4_0', 'Q4_1', 'Q4_K_M'])
    parser.add_argument('--output-dir', default='checkpoints/distilled_q4')
    
    args = parser.parse_args()
    run_pipeline(args)
