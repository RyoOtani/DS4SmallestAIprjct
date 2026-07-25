#!/usr/bin/env python3
"""
distill.py — Knowledge Distillation for TinyLLM

Distills knowledge from a large teacher model (DeepSeek V4, Qwen, etc.)
to a smaller TinyLLM student model.

Methods:
  - Logit-level distillation (KL divergence between teacher/student logits)
  - Hidden-state distillation (MSE between intermediate representations)
  - Attention-map distillation (MSE between attention patterns)

Usage:
  python distill.py --teacher deepseek-ai/DeepSeek-V4 --student checkpoints/nano/model.pt
"""

import os, sys, json, math, time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class DistillConfig:
    """Distillation hyperparameters."""
    # Temperature
    temperature: float = 3.0         # Higher = softer probability distribution
    
    # Loss weights
    kd_loss_weight: float = 0.7      # Weight for KL divergence loss
    ce_loss_weight: float = 0.3      # Weight for standard cross-entropy (ground truth)
    hidden_loss_weight: float = 0.1  # Weight for hidden state MSE (optional)
    attn_loss_weight: float = 0.05   # Weight for attention map MSE (optional)
    
    # Training
    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 10000
    max_seq_len: int = 1024
    log_interval: int = 50
    save_interval: int = 1000
    
    # Data
    data_path: str = "data/train.bin"
    val_data_path: Optional[str] = "data/val.bin"
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    
    # Multi-GPU
    use_ddp: bool = False


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.
    
    L = α * L_KD + β * L_CE + γ * L_hidden + δ * L_attn
    
    where:
      L_KD = KL(softmax(teacher_logits/T) || softmax(student_logits/T)) * T²
      L_CE = CrossEntropy(student_logits, labels)
      L_hidden = MSE(student_hidden, teacher_hidden_projection)
      L_attn = MSE(student_attn, teacher_attn)
    """
    
    def __init__(self, config: DistillConfig):
        super().__init__()
        self.cfg = config
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        student_attn: Optional[torch.Tensor] = None,
        teacher_attn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: [B, S, V] student model logits
            teacher_logits: [B, S, V] teacher model logits
            labels: [B, S] ground truth token IDs
            student_hidden: [B, S, D] student last hidden state
            teacher_hidden: [B, S, D_teacher] teacher last hidden state
            student_attn: [B, H, S, S] student attention maps (optional)
            teacher_attn: [B, H, S, S] teacher attention maps (optional)
        
        Returns:
            (total_loss, loss_components_dict)
        """
        T = self.cfg.temperature
        
        # 1. KD loss (KL divergence)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # Flatten to [B*S, V]
        B, S, V = student_logits.shape
        student_log_probs_flat = student_log_probs.view(-1, V)
        teacher_probs_flat = teacher_probs.view(-1, V)
        
        kd_loss = self.kl_div(student_log_probs_flat, teacher_probs_flat) * (T ** 2)
        
        # 2. CE loss (ground truth)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, V),
            labels.view(-1),
            ignore_index=-100,
        )
        
        total_loss = (self.cfg.kd_loss_weight * kd_loss +
                      self.cfg.ce_loss_weight * ce_loss)
        
        components = {'kd_loss': kd_loss.item(), 'ce_loss': ce_loss.item()}
        
        # 3. Hidden state loss (optional)
        if student_hidden is not None and teacher_hidden is not None:
            # Project teacher hidden to student hidden dim if needed
            if teacher_hidden.shape[-1] != student_hidden.shape[-1]:
                # Use a simple linear projection (created outside, passed in)
                pass
            hidden_loss = self.mse(student_hidden, teacher_hidden)
            total_loss = total_loss + self.cfg.hidden_loss_weight * hidden_loss
            components['hidden_loss'] = hidden_loss.item()
        
        # 4. Attention loss (optional)
        if student_attn is not None and teacher_attn is not None:
            attn_loss = self.mse(student_attn, teacher_attn)
            total_loss = total_loss + self.cfg.attn_loss_weight * attn_loss
            components['attn_loss'] = attn_loss.item()
        
        return total_loss, components


class DistillTrainer:
    """
    Knowledge distillation trainer.
    
    Supports:
      - Offline distillation (pre-computed teacher logits on disk)
      - Online distillation (teacher and student run simultaneously)
      - Mixed precision training
      - Gradient accumulation
      - Checkpoint saving/resuming
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module],
        tokenizer,
        config: DistillConfig,
        device: torch.device = None,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.student = self.student.to(self.device)
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        
        self.loss_fn = DistillationLoss(config)
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.global_step = 0
        
    def setup_optimizer(self):
        """Initialize optimizer and LR scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        
        def lr_lambda(step):
            if step < self.cfg.warmup_steps:
                return step / max(1, self.cfg.warmup_steps)
            progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.max_steps - self.cfg.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Get teacher logits
        with torch.no_grad():
            if self.teacher is not None:
                teacher_out = self.teacher(input_ids=input_ids)
                teacher_logits = teacher_out['logits'].detach()
            else:
                # Offline mode: teacher logits should be in batch
                teacher_logits = batch['teacher_logits'].to(self.device)
        
        # Forward student
        amp_ctx = torch.cuda.amp.autocast(dtype=self.cfg.amp_dtype) if self.cfg.use_amp else torch.no_grad()
        with amp_ctx:
            student_out = self.student(input_ids=input_ids)
            student_logits = student_out['logits']
            
            loss, components = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
            )
            loss = loss / self.cfg.grad_accum_steps
        
        # Backward
        if self.cfg.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {**components, 'loss': loss.item() * self.cfg.grad_accum_steps}
    
    def optimizer_step(self):
        """Step optimizer with gradient clipping."""
        if self.cfg.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def train(self, dataloader: DataLoader, resume_from: int = 0):
        """Main training loop."""
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.student.train()
        self.global_step = resume_from
        
        data_iter = iter(dataloader)
        total_loss = 0.0
        total_kd = 0.0
        total_ce = 0.0
        
        pbar = tqdm(total=self.cfg.max_steps - resume_from, desc='Distilling')
        start_time = time.time()
        
        for _ in range(resume_from, self.cfg.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            total_kd += metrics.get('kd_loss', 0)
            total_ce += metrics.get('ce_loss', 0)
            
            if (self.global_step + 1) % self.cfg.grad_accum_steps == 0:
                self.optimizer_step()
            
            self.global_step += 1
            pbar.update(1)
            
            if self.global_step % self.cfg.log_interval == 0:
                avg_loss = total_loss / self.cfg.log_interval
                avg_kd = total_kd / self.cfg.log_interval
                avg_ce = total_ce / self.cfg.log_interval
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'kd': f'{avg_kd:.4f}',
                    'ce': f'{avg_ce:.4f}',
                    'lr': f'{lr:.2e}',
                })
                total_loss = total_kd = total_ce = 0.0
            
            if self.global_step % self.cfg.save_interval == 0:
                self.save_checkpoint()
        
        pbar.close()
        elapsed = time.time() - start_time
        print(f"\n✅ Distillation complete! {elapsed:.0f}s, {self.global_step} steps")
        self.save_checkpoint(final=True)
    
    def save_checkpoint(self, final: bool = False):
        """Save student model checkpoint."""
        ckpt_dir = f'checkpoints/distill_step_{self.global_step}' if not final else 'checkpoints/distill_final'
        os.makedirs(ckpt_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.cfg,
        }, f'{ckpt_dir}/model.pt')
        
        if final:
            self.tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 Final model saved to {ckpt_dir}/")


# ═══════════════════════════════════════════════════════════════
# Offline Distillation Dataset
# ═══════════════════════════════════════════════════════════════

class OfflineDistillDataset(Dataset):
    """
    Dataset for offline distillation: pre-computed teacher logits stored on disk.
    
    File format (binary):
      [num_samples: u32]
      For each sample:
        [input_len: u32] [input_ids: u32[input_len]]
        [logit_len: u32] [logits: f16[logit_len * vocab_size]]  # flattened
    """
    
    def __init__(self, data_path: str, seq_len: int, vocab_size: int):
        self.data_path = data_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._load_index()
    
    def _load_index(self):
        """Build index of sample offsets."""
        self.offsets = []
        with open(self.data_path, 'rb') as f:
            num_samples = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_samples):
                self.offsets.append(f.tell())
                input_len = struct.unpack('<I', f.read(4))[0]
                f.seek(input_len * 4, 1)  # skip input_ids
                logit_len = struct.unpack('<I', f.read(4))[0]
                f.seek(logit_len * self.vocab_size * 2, 1)  # skip logits (f16)
        self.num_samples = len(self.offsets)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with open(self.data_path, 'rb') as f:
            f.seek(self.offsets[idx])
            input_len = struct.unpack('<I', f.read(4))[0]
            input_ids = np.frombuffer(f.read(input_len * 4), dtype=np.int32)
            logit_len = struct.unpack('<I', f.read(4))[0]
            logits = np.frombuffer(f.read(logit_len * self.vocab_size * 2), dtype=np.float16)
            logits = logits.reshape(logit_len, self.vocab_size)
        
        # Pad/truncate to seq_len
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
            logits = logits[:self.seq_len]
        
        labels = np.roll(input_ids, -1)
        labels[-1] = -100  # mask last position
        
        return {
            'input_ids': torch.from_numpy(input_ids.astype(np.int64)),
            'labels': torch.from_numpy(labels.astype(np.int64)),
            'teacher_logits': torch.from_numpy(logits.astype(np.float32)),
        }


# ═══════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Testing distillation loss...")
    
    cfg = DistillConfig(temperature=3.0)
    loss_fn = DistillationLoss(cfg)
    
    B, S, V = 2, 8, 1000
    student_logits = torch.randn(B, S, V, requires_grad=True)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    
    loss, comps = loss_fn(student_logits, teacher_logits, labels)
    print(f"  KD loss: {comps['kd_loss']:.4f}")
    print(f"  CE loss: {comps['ce_loss']:.4f}")
    print(f"  Total:   {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print(f"  Grad norm: {student_logits.grad.norm().item():.4f}")
    
    print("✅ Distillation loss test passed")
