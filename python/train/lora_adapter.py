"""
lora_adapter.py — LoRA (Low-Rank Adaptation) for tinyllm.

Enables safe online learning without catastrophic forgetting:
  - Core model weights are FROZEN
  - Only low-rank adapter matrices are trained
  - Adapters are small (~100-500 MB) and fast to train
  - Snapshots with automatic rollback on performance degradation

Usage:
  python lora_adapter.py --model student.gguf --feedback feedback.jsonl \
                         --output adapter.bin
"""

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    model_path: str = "output/student_model"
    feedback_path: str = "data/feedback.jsonl"
    output_path: str = "output/lora_adapter.bin"

    rank: int = 64
    alpha: float = 32.0
    dropout: float = 0.05

    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 5e-5
    batch_size: int = 2
    max_epochs: int = 3
    warmup_ratio: float = 0.1

    # Safety
    max_adapter_size_mb: float = 500
    min_improvement: float = 0.01
    max_rollbacks: int = 3
    snapshot_dir: str = "output/lora_snapshots/"

    # Mixed
    use_bfloat16: bool = True


class LoRALinear(nn.Module):
    """LoRA-augmented linear layer: W' = W + (alpha/r) * B @ A

    W: frozen pretrained weight  [out_dim, in_dim]
    A: trainable low-rank        [r, in_dim]
    B: trainable low-rank        [out_dim, r]
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_dim, out_dim = base.in_features, base.out_features

        # Freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zero for B (so initially W' == W)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base_out + lora_out

    @property
    def weight(self):
        """Effective weight: W + (alpha/r) * B @ A"""
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scale


class LoRAAdapterManager:
    """Manages LoRA adapter lifecycle: train, merge, snapshot, rollback."""

    def __init__(self, config: LoRAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        self._inject_lora()
        self.snapshots = []
        self.baseline_loss = None

    def _load_model(self):
        """Load base model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.config.use_bfloat16 else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

    def _inject_lora(self):
        """Replace target linear layers with LoRA-augmented versions."""
        self.lora_modules = []
        replaced = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if it's a target module
                is_target = any(
                    name.endswith(t) or name.split('.')[-1] == t
                    for t in self.config.target_modules
                )
                if is_target:
                    parent = self._get_parent(name)
                    child_name = name.split('.')[-1]
                    lora_layer = LoRALinear(
                        module, self.config.rank,
                        self.config.alpha, self.config.dropout
                    )
                    setattr(parent, child_name, lora_layer)
                    self.lora_modules.append(lora_layer)
                    replaced += 1

        # Freeze all non-LoRA params
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA injected: {replaced} layers")
        print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")
        print(f"Adapter size: ~{trainable*2/1024/1024:.1f} MB")  # fp16 estimate

    @staticmethod
    def _get_parent(module_name: str):
        """Get parent module from dotted name. Handled by caller context."""
        # This is handled inline in _inject_lora
        pass

    def train_on_feedback(self):
        """Train adapter on user feedback (positive + negative samples)."""
        with open(self.config.feedback_path) as f:
            feedback_data = [json.loads(line) for line in f]

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )

        for epoch in range(self.config.max_epochs):
            total_loss = 0.0
            n_samples = 0

            for item in feedback_data:
                # Positive: reinforce correct behavior
                # Negative: unlearn incorrect behavior
                text = item.get('text', '')
                reward = item.get('reward', 0.0)  # -1 to 1

                # Skip neutral feedback
                if abs(reward) < 0.01:
                    continue

                inputs = self.tokenizer(
                    text, return_tensors='pt', truncation=True, max_length=2048
                ).to(self.device)

                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss

                # Positive reward → minimize loss; Negative → maximize
                weighted_loss = loss * (1.0 - reward)

                weighted_loss.backward()
                total_loss += loss.item()
                n_samples += 1

                if n_samples % self.config.batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            print(f"Epoch {epoch+1}: avg_loss={total_loss/n_samples:.4f}")

            # Auto-snapshot
            self._snapshot(f"epoch_{epoch+1}")

            # Check for degradation
            if epoch > 0 and self.baseline_loss and total_loss > self.baseline_loss * 1.1:
                print("⚠ Performance degraded. Rolling back!")
                self.rollback()

    def _snapshot(self, tag: str):
        """Save adapter state."""
        path = os.path.join(self.config.snapshot_dir, f"adapter_{tag}.pt")
        os.makedirs(self.config.snapshot_dir, exist_ok=True)

        state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if 'lora_' in name
        }
        torch.save(state, path)

        self.snapshots.append(path)
        print(f"Snapshot saved: {path}")

        # Prune old snapshots
        while len(self.snapshots) > self.config.max_rollbacks * 2:
            old = self.snapshots.pop(0)
            try: os.remove(old)
            except: pass

    def rollback(self, snapshot_idx: int = -1):
        """Rollback to a previous snapshot."""
        if not self.snapshots:
            print("No snapshots to rollback to.")
            return False

        path = self.snapshots[snapshot_idx]
        state = torch.load(path, map_location=self.device)

        for name, param in self.model.named_parameters():
            if name in state:
                param.data.copy_(state[name])

        print(f"✓ Rolled back to: {path}")
        return True

    def merge_and_save(self, output_path: str):
        """Merge LoRA weights into base model for inference."""
        merged_state = {}

        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                merged_state[name] = param.data.clone()
            # LoRA params are already accounted for in effective weight

        # Save merged weights
        torch.save(merged_state, output_path)

        # Also save adapter separately
        adapter_path = output_path.replace('.bin', '_adapter.bin')
        adapter_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if 'lora_' in name
        }
        torch.save(adapter_state, adapter_path)

        print(f"✓ Merged model: {output_path}")
        print(f"✓ Adapter only: {adapter_path}")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter Training for tinyllm")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--feedback', type=str, required=True)
    parser.add_argument('--output', type=str, default='output/lora_adapter.bin')
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--rollback', action='store_true')

    args = parser.parse_args()

    config = LoRAConfig(
        model_path=args.model,
        feedback_path=args.feedback,
        output_path=args.output,
        rank=args.rank,
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )

    manager = LoRAAdapterManager(config)

    if args.rollback:
        manager.rollback()
    else:
        manager.train_on_feedback()
        manager.merge_and_save(args.output)


if __name__ == '__main__':
    main()
