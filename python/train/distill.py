"""
distill.py — Knowledge Distillation: teacher → student

Transfers knowledge from a large teacher model (e.g., DeepSeek-V3, GPT-4)
to a compact student model suitable for tinyllm.

Strategies:
  1. Logit-level KD: student learns to match teacher's output distribution
  2. Hidden-state KD: match intermediate layer representations
  3. Multi-Token Prediction (MTP): predict next N tokens, not just 1
  4. FIM (Fill-in-the-Middle): train on code infilling tasks

Usage:
  python distill.py --teacher deepseek-v3 --student qwen3-5-9b \
                    --data code_dataset.jsonl --output student_merged.gguf
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm


@dataclass
class DistillConfig:
    """Distillation hyperparameters."""
    teacher_model: str = "deepseek-ai/DeepSeek-V3"
    student_model: str = "Qwen/Qwen3-5-9B"
    data_path: str = "data/train.jsonl"
    output_path: str = "output/distilled_model"
    max_seq_len: int = 8192
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    max_steps: int = 10000

    # KD loss weights
    kd_temperature: float = 3.0      # soften teacher logits
    alpha_logit: float = 0.7         # weight for logit KD
    alpha_hidden: float = 0.2        # weight for hidden-state KD
    alpha_mtp: float = 0.1           # weight for multi-token prediction

    # MTP
    mtp_depth: int = 2               # predict next N tokens
    mtp_heads: int = 4               # parallel prediction heads

    # FIM (Fill-in-the-Middle)
    fim_rate: float = 0.5            # probability of FIM formatting

    # Quantization
    quantize: bool = True
    q_bits: int = 4
    mixed_precision: bool = True     # 6-bit for key layers, 4-bit for rest

    # LoRA
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: float = 16.0

    # Optimization
    use_fp8: bool = False
    compile_model: bool = True
    gradient_checkpointing: bool = True


class MultiTokenPredictionHead(nn.Module):
    """MTP: Predict next N tokens in parallel.

    Each head h_k predicts token at position t+k given hidden state at position t.
    Uses independent output projections (or shared with depth-dependent bias).
    """

    def __init__(self, hidden_dim: int, vocab_size: int, depth: int):
        super().__init__()
        self.depth = depth
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size, bias=False),
            )
            for _ in range(depth)
        ])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Return list of logits for each future position."""
        return [head(hidden) for head in self.heads]


class DistillationTrainer:
    """Handles the teacher-student distillation process."""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self._load_models()
        self._setup_optimizer()

    def _load_models(self):
        """Load teacher and student models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading teacher: {self.config.teacher_model}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        print(f"Loading student: {self.config.student_model}")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.config.student_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model, trust_remote_code=True
        )

        # Add FIM special tokens if not present
        fim_tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
        for tok in fim_tokens:
            if tok not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([tok])

        self.student.resize_token_embeddings(len(self.tokenizer))

        # MTP heads
        if self.config.alpha_mtp > 0:
            hidden_dim = self.student.config.hidden_size
            vocab_size = len(self.tokenizer)
            self.mtp_head = MultiTokenPredictionHead(
                hidden_dim, vocab_size, self.config.mtp_depth
            ).to(self.device)

        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()

        # torch.compile for speed
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling student model...")
            self.student = torch.compile(self.student)

        if self.config.use_fp8:
            self.student = self.student.to(torch.float8_e4m3fn)

    def _setup_optimizer(self):
        params = list(self.student.parameters())
        if hasattr(self, 'mtp_head'):
            params += list(self.mtp_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, fused=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.max_steps
        )

    def _fim_format(self, text: str) -> tuple[str, str, str]:
        """Randomly split text for FIM: (prefix, middle, suffix)."""
        import random
        tokens = text.split()
        if len(tokens) < 10:
            return text, "", ""

        # Pick random split points
        prefix_end = random.randint(len(tokens) // 4, len(tokens) // 2)
        suffix_start = random.randint(prefix_end + 2, len(tokens) * 3 // 4)

        prefix = " ".join(tokens[:prefix_end])
        middle = " ".join(tokens[prefix_end:suffix_start])
        suffix = " ".join(tokens[suffix_start:])

        return prefix, middle, suffix

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute distillation losses."""
        T = self.config.kd_temperature
        mask = attention_mask[:, 1:].bool()  # shift for next-token

        losses = {}

        # 1. Standard LM loss (cross-entropy with labels)
        lm_loss = F.cross_entropy(
            student_logits[mask].float(),
            labels[mask],
            reduction='mean',
        )
        losses['lm_loss'] = lm_loss

        # 2. Logit-level KD (soft targets from teacher)
        if self.config.alpha_logit > 0:
            soft_student = F.log_softmax(student_logits[mask].float() / T, dim=-1)
            soft_teacher = F.softmax(teacher_logits[mask].float() / T, dim=-1)
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
            losses['kd_loss'] = kd_loss

        # 3. Hidden-state KD (optional, needs teacher hidden states)
        if self.config.alpha_hidden > 0 and hasattr(self, '_teacher_hidden'):
            student_hidden = getattr(self, '_student_hidden', None)
            teacher_hidden = getattr(self, '_teacher_hidden', None)
            if student_hidden is not None and teacher_hidden is not None:
                hidden_loss = F.mse_loss(student_hidden.float(), teacher_hidden.float())
                losses['hidden_loss'] = hidden_loss

        # Total
        total = lm_loss
        if 'kd_loss' in losses:
            total = total + self.config.alpha_logit * losses['kd_loss']
        if 'hidden_loss' in losses:
            total = total + self.config.alpha_hidden * losses['hidden_loss']

        losses['total'] = total
        return losses

    def _compute_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-Token Prediction loss: predict next N tokens."""
        losses = []
        depth = self.config.mtp_depth

        for k in range(1, depth + 1):
            # Target: tokens shifted by k positions
            targets = input_ids[:, k:]
            mask = attention_mask[:, k:].bool()

            if mask.sum() == 0:
                continue

            # Predictions from MTP head k-1
            preds = self.mtp_head.heads[k - 1](hidden_states[:, :-k]) if k > 1 else hidden_states[:, :-1]
            if k == 1:
                preds = self.student.lm_head(preds)
            else:
                # MTP head already produces vocab-sized logits
                pass

            # Ensure alignment
            min_len = min(preds.size(1), targets.size(1))
            preds = preds[:, :min_len, :]
            targets = targets[:, :min_len]
            mask = mask[:, :min_len]

            loss = F.cross_entropy(
                preds[mask].float(),
                targets[mask],
                reduction='mean',
            )
            losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch.get('labels', input_ids.clone())

        # Shift labels for next-token prediction
        labels = labels[:, 1:].contiguous()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.config.alpha_hidden > 0,
            )
            teacher_logits = teacher_out.logits[:, :-1, :]

        # Student forward
        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = student_out.logits[:, :-1, :]

        # KD losses
        losses = self._compute_kd_loss(
            student_logits, teacher_logits, labels, attention_mask
        )

        # MTP loss
        if self.config.alpha_mtp > 0:
            mtp_loss = self._compute_mtp_loss(
                student_out.hidden_states[-1],
                input_ids,
                attention_mask,
            )
            losses['mtp_loss'] = mtp_loss
            losses['total'] = losses['total'] + self.config.alpha_mtp * mtp_loss

        # Backward
        loss = losses['total'] / self.config.gradient_accumulation
        loss.backward()
        return {k: v.item() for k, v in losses.items()}

    def train(self):
        """Main training loop."""
        from torch.utils.data import DataLoader

        # Load dataset
        with open(self.config.data_path) as f:
            data = [json.loads(line) for line in f]

        dataloader = DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        global_step = 0
        pbar = tqdm(total=self.config.max_steps, desc="Distilling")

        while global_step < self.config.max_steps:
            for batch in dataloader:
                metrics = self.train_step(batch)

                global_step += 1
                if global_step % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                pbar.update(1)
                pbar.set_postfix(metrics)

                if global_step >= self.config.max_steps:
                    break

                # Save checkpoint periodically
                if global_step % 1000 == 0:
                    self._save_checkpoint(global_step)

        pbar.close()
        self._save_final()

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Tokenize and collate a batch."""
        texts = []
        for item in batch:
            text = item.get('text', item.get('content', ''))
            if self.config.fim_rate > 0 and torch.rand(1).item() < self.config.fim_rate:
                prefix, middle, suffix = self._fim_format(text)
                text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"

            texts.append(text)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors='pt',
        )
        return encoded

    def _save_checkpoint(self, step: int):
        path = f"{self.config.output_path}_step{step}"
        os.makedirs(path, exist_ok=True)
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _save_final(self):
        """Save final model, optionally quantized."""
        path = self.config.output_path
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        if self.config.quantize:
            print(f"Quantizing to {self.config.q_bits}-bit and exporting GGUF...")
            from pathlib import Path
            gguf_path = Path(path).with_suffix('.gguf')

            # Use llama.cpp's convert script if available
            import subprocess
            cmd = [
                "python", "-m", "llama_cpp.convert",
                "--src", path,
                "--dst", str(gguf_path),
                "--qbits", str(self.config.q_bits),
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ GGUF exported: {gguf_path}")
            except Exception as e:
                print(f"⚠ GGUF conversion skipped: {e}")
                print("  Manually convert with: python llama.cpp/convert.py")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for tinyllm")
    parser.add_argument('--teacher', type=str, default='deepseek-ai/DeepSeek-V3')
    parser.add_argument('--student', type=str, default='Qwen/Qwen3-5-9B')
    parser.add_argument('--data', type=str, default='data/train.jsonl')
    parser.add_argument('--output', type=str, default='output/student_model')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--fim-rate', type=float, default=0.5)
    parser.add_argument('--mtp-depth', type=int, default=2)
    parser.add_argument('--quantize', action='store_true', default=True)
    parser.add_argument('--no-quantize', dest='quantize', action='store_false')
    parser.add_argument('--lora', action='store_true', default=False)

    args = parser.parse_args()

    config = DistillConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        max_steps=args.steps,
        learning_rate=args.lr,
        kd_temperature=args.temperature,
        fim_rate=args.fim_rate,
        mtp_depth=args.mtp_depth,
        quantize=args.quantize,
        use_lora=args.lora,
    )

    trainer = DistillationTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
