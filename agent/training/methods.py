"""
Training Methods Collection — Multiple ways to train TinyLLM.

Methods:
  1. Full Fine-tuning       — Train all parameters (FSDP/DeepSpeed)
  2. LoRA / QLoRA           — Low-Rank Adaptation (efficient)
  3. DPO                    — Direct Preference Optimization (RLHF alternative)
  4. Instruction Tuning     — Supervised fine-tuning on instruction datasets
  5. Knowledge Distillation — Train small model from large teacher
  6. Continued Pre-training — Extend base model on domain data
  7. Multi-task Learning    — Joint training on multiple objectives
  8. Curriculum Learning    — Progressive difficulty increase

Each method is a self-contained training strategy with presets for common use cases.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal


# ══════════════════════════════════════════════════════════════════════════════
# Shared Types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingPreset:
    """Pre-configured training preset."""
    name: str
    method: str                     # lora, full, dpo, distill, instruct, domain, multitask, curriculum
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_len: int = 2048
    description: str = ""


@dataclass
class TrainingResult:
    """Result of a training run."""
    method: str
    train_loss: list[float] = field(default_factory=list)
    eval_loss: list[float] = field(default_factory=list)
    steps: int = 0
    duration_s: float = 0.0
    best_metric: float = 0.0
    checkpoint_path: str = ""
    success: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Presets
# ══════════════════════════════════════════════════════════════════════════════

TRAINING_PRESETS: dict[str, TrainingPreset] = {
    # ── LoRA fine-tuning ──────────────────────────────────────────────────
    "lora-fast": TrainingPreset(
        name="lora-fast", method="lora",
        learning_rate=2e-4, batch_size=16, epochs=1,
        description="Quick LoRA fine-tune (1 epoch, 30min on M2)",
    ),
    "lora-standard": TrainingPreset(
        name="lora-standard", method="lora",
        learning_rate=1e-4, batch_size=8, epochs=3,
        description="Standard LoRA fine-tuning (3 epochs)",
    ),
    "lora-deep": TrainingPreset(
        name="lora-deep", method="lora",
        learning_rate=5e-5, batch_size=4, epochs=10,
        description="Deep LoRA fine-tuning (10 epochs, best quality)",
    ),

    # ── QLoRA (4-bit quantization + LoRA) ────────────────────────────────
    "qlora-4bit": TrainingPreset(
        name="qlora-4bit", method="qlora",
        learning_rate=2e-4, batch_size=8, epochs=3,
        description="QLoRA 4-bit — fits 14B model in 8GB RAM",
    ),

    # ── Full fine-tuning ─────────────────────────────────────────────────
    "full-quick": TrainingPreset(
        name="full-quick", method="full",
        learning_rate=5e-5, batch_size=4, epochs=1,
        description="Quick full fine-tune (A100×4+)",
    ),
    "full-standard": TrainingPreset(
        name="full-standard", method="full",
        learning_rate=2e-5, batch_size=2, epochs=3,
        description="Standard full fine-tune (A100×8+)",
    ),

    # ── DPO (Direct Preference Optimization) ─────────────────────────────
    "dpo-standard": TrainingPreset(
        name="dpo-standard", method="dpo",
        learning_rate=5e-5, batch_size=4, epochs=1,
        description="DPO alignment training (preference pairs)",
    ),
    "dpo-deep": TrainingPreset(
        name="dpo-deep", method="dpo",
        learning_rate=2e-5, batch_size=2, epochs=3,
        description="Deep DPO for stronger alignment",
    ),

    # ── Knowledge Distillation ───────────────────────────────────────────
    "distill-soft": TrainingPreset(
        name="distill-soft", method="distill",
        learning_rate=1e-4, batch_size=8, epochs=3,
        description="Soft-label distillation from teacher to student",
    ),
    "distill-hard": TrainingPreset(
        name="distill-hard", method="distill",
        learning_rate=5e-5, batch_size=4, epochs=5,
        description="Hard-label + feature distillation",
    ),

    # ── Instruction Tuning ───────────────────────────────────────────────
    "instruct-fast": TrainingPreset(
        name="instruct-fast", method="instruct",
        learning_rate=2e-4, batch_size=16, epochs=1,
        description="Quick instruction tuning (ShareGPT/Alpaca style)",
    ),
    "instruct-standard": TrainingPreset(
        name="instruct-standard", method="instruct",
        learning_rate=1e-4, batch_size=8, epochs=3,
        description="Standard instruction tuning",
    ),

    # ── Domain Continued Pre-training ────────────────────────────────────
    "domain-code": TrainingPreset(
        name="domain-code", method="domain",
        learning_rate=5e-5, batch_size=4, epochs=1,
        max_seq_len=4096,
        description="Domain adaptation on code (The Stack, GitHub)",
    ),
    "domain-japanese": TrainingPreset(
        name="domain-japanese", method="domain",
        learning_rate=5e-5, batch_size=4, epochs=1,
        max_seq_len=2048,
        description="Domain adaptation on Japanese text",
    ),

    # ── Curriculum Learning ──────────────────────────────────────────────
    "curriculum-easy": TrainingPreset(
        name="curriculum-easy", method="curriculum",
        learning_rate=1e-4, batch_size=8, epochs=1,
        description="Phase 1: easy examples only",
    ),
    "curriculum-full": TrainingPreset(
        name="curriculum-full", method="curriculum",
        learning_rate=5e-5, batch_size=4, epochs=3,
        description="Phase 2: mixed difficulty (after curriculum-easy)",
    ),

    # ── Multi-task ───────────────────────────────────────────────────────
    "multitask-standard": TrainingPreset(
        name="multitask-standard", method="multitask",
        learning_rate=1e-4, batch_size=8, epochs=3,
        description="Joint training: code + chat + reasoning",
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Training Method Implementations
# ══════════════════════════════════════════════════════════════════════════════

class LoRATrainer:
    """
    Low-Rank Adaptation (LoRA) fine-tuning.

    Freezes base model, trains only small rank-decomposition matrices.
    Memory: ~1% of full fine-tuning.
    Quality: 95-99% of full fine-tuning for most tasks.
    """

    DEFAULT_RANK = 16
    DEFAULT_ALPHA = 32
    DEFAULT_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj", "o_proj"]  # attention only

    @staticmethod
    def apply_lora(model, rank: int = 16, alpha: int = 32,
                   dropout: float = 0.05, target_modules: list[str] = None):
        """Apply LoRA adapters to a model. Returns (model, trainable_params)."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("⚠️  peft not installed. Install: pip install peft")
            return model, sum(p.numel() for p in model.parameters())

        target = target_modules or LoRATrainer.TARGET_MODULES

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target,
            bias="none",
        )

        model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"🧩 LoRA applied: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)")
        return model, trainable

    @staticmethod
    def merge_and_unload(model):
        """Merge LoRA weights into base model and unload adapters."""
        try:
            return model.merge_and_unload()
        except AttributeError:
            print("⚠️  Model does not support merge_and_unload (not a PEFT model)")
            return model


class QLoRATrainer:
    """
    QLoRA — 4-bit quantized base model + LoRA adapters.

    Memory: Fits 14B model in 8GB, 70B in 32GB.
    Quality: Within 1-2% of full 16-bit LoRA.
    """

    @staticmethod
    def apply_qlora(model, rank: int = 16, alpha: int = 32,
                    bits: int = 4, double_quant: bool = True):
        """Apply QLoRA (4-bit quantization + LoRA)."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            import torch
        except ImportError:
            print("⚠️  peft + bitsandbytes required. Install: pip install peft bitsandbytes")
            return model, 0

        # Prepare for 4-bit training
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
            bias="none",
        )

        model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🧩 QLoRA ({bits}-bit) applied: {trainable:,} trainable params")
        return model, trainable


class DPOTrainer:
    """
    Direct Preference Optimization — align model to human preferences.

    Unlike RLHF, DPO doesn't need a separate reward model.
    Input: (prompt, chosen_response, rejected_response) triplets.
    Reference: Rafailov et al. 2023, "Direct Preference Optimization"
    """

    @staticmethod
    def prepare_dataset(
        prompts: list[str],
        chosen: list[str],
        rejected: list[str],
    ) -> list[dict]:
        """Prepare DPO training data."""
        return [
            {"prompt": p, "chosen": c, "rejected": r}
            for p, c, r in zip(prompts, chosen, rejected)
            if p and c and r
        ]

    @staticmethod
    def dpo_loss(
        model,
        ref_model,
        batch: dict,
        beta: float = 0.1,
    ) -> tuple:
        """
        Compute DPO loss.

        L_DPO = -log(σ(β * (log π_θ(chosen|p) - log π_ref(chosen|p))
                          - β * (log π_θ(rejected|p) - log π_ref(rejected|p))))
        """
        import torch
        import torch.nn.functional as F

        # Get policy model log-probs
        with torch.no_grad():
            ref_chosen_logps = DPOTrainer._get_logps(ref_model, batch["prompt"], batch["chosen"])
            ref_rejected_logps = DPOTrainer._get_logps(ref_model, batch["prompt"], batch["rejected"])

        policy_chosen_logps = DPOTrainer._get_logps(model, batch["prompt"], batch["chosen"])
        policy_rejected_logps = DPOTrainer._get_logps(model, batch["prompt"], batch["rejected"])

        # DPO objective
        chosen_diff = policy_chosen_logps - ref_chosen_logps
        rejected_diff = policy_rejected_logps - ref_rejected_logps
        logits = beta * (chosen_diff - rejected_diff)
        loss = -F.logsigmoid(logits).mean()

        # Accuracy metric
        accuracy = (logits > 0).float().mean()

        return loss, accuracy

    @staticmethod
    def _get_logps(model, prompts, responses):
        """Get log-probabilities of responses given prompts."""
        import torch
        # Placeholder: concatenate prompt+response, compute log-probs
        combined = [p + r for p, r in zip(prompts, responses)]
        # In practice: use model(combined).logits → log_softmax → gather token log-probs
        # Simplified for now
        return torch.tensor(0.0, requires_grad=True)


class DistillationTrainer:
    """
    Knowledge Distillation — train a small student model from a large teacher.

    Methods:
      - Soft distillation: match teacher's output probabilities
      - Hard distillation: match teacher's argmax predictions
      - Feature distillation: match intermediate layer outputs
    """

    @staticmethod
    def distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature: float = 3.0,
        alpha: float = 0.5,  # weight between distillation and hard-label loss
    ):
        """
        Combined distillation + cross-entropy loss.

        L = α * T² * KL(softmax(teacher/T) || softmax(student/T))
          + (1-α) * CE(student, labels)
        """
        import torch
        import torch.nn.functional as F

        # Soft distillation loss
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        distill_loss *= temperature * temperature

        # Hard-label loss
        ce_loss = F.cross_entropy(student_logits, labels)

        return alpha * distill_loss + (1 - alpha) * ce_loss

    @staticmethod
    def feature_distillation_loss(
        student_features: list,
        teacher_features: list,
    ):
        """MSE loss between student and teacher intermediate features."""
        import torch
        import torch.nn.functional as F
        loss = 0.0
        for sf, tf in zip(student_features, teacher_features):
            loss += F.mse_loss(sf, tf.detach())
        return loss / len(student_features)


class InstructionTuner:
    """
    Instruction Tuning — supervised fine-tuning on instruction-following data.

    Formats:
      - Alpaca: ### Instruction: ... ### Response: ...
      - ShareGPT: {"messages": [{"role":"user",...}, {"role":"assistant",...}]}
      - ChatML: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
    """

    FORMATS = ["alpaca", "sharegpt", "chatml", "tinyllm"]

    @staticmethod
    def format_alpaca(instruction: str, input_text: str = "", output: str = "") -> str:
        """Format instruction in Alpaca style."""
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    @staticmethod
    def format_chatml(messages: list[dict]) -> str:
        """Format as ChatML."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)

    @staticmethod
    def format_tinyllm(messages: list[dict]) -> str:
        """TinyLLM native format (simple, efficient)."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = {"user": "User:", "assistant": "Assistant:", "system": "System:"}
            parts.append(f"{prefix.get(role, role)} {content}")
        return "\n".join(parts)

    @staticmethod
    def load_dataset(
        dataset_path: str,
        format: str = "sharegpt",
        max_samples: int = 10000,
    ) -> list[dict]:
        """Load and format an instruction dataset."""
        import json

        with open(dataset_path) as f:
            data = json.load(f)

        formatted = []
        for item in data[:max_samples]:
            if format == "sharegpt":
                text = InstructionTuner.format_chatml(item.get("messages", []))
            elif format == "alpaca":
                text = InstructionTuner.format_alpaca(
                    item.get("instruction", ""),
                    item.get("input", ""),
                    item.get("output", ""),
                )
            elif format == "chatml":
                text = InstructionTuner.format_chatml(item.get("messages", []))
            else:
                text = InstructionTuner.format_tinyllm(item.get("messages", []))

            formatted.append({"text": text})

        print(f"📚 Loaded {len(formatted)} instruction samples ({format} format)")
        return formatted


# ══════════════════════════════════════════════════════════════════════════════
# Training Method Registry
# ══════════════════════════════════════════════════════════════════════════════

class TrainingMethodRegistry:
    """Registry of all available training methods."""

    METHODS = {
        "lora": {
            "name": "LoRA Fine-tuning",
            "description": "Low-Rank Adaptation — efficient fine-tuning",
            "memory": "Low (1-10% of full FT)",
            "quality": "95-99% of full FT",
            "presets": ["lora-fast", "lora-standard", "lora-deep"],
            "trainer": LoRATrainer,
        },
        "qlora": {
            "name": "QLoRA (4-bit)",
            "description": "4-bit quantized LoRA — fits large models in small RAM",
            "memory": "Very Low (14B in 8GB)",
            "quality": "93-98% of full FT",
            "presets": ["qlora-4bit"],
            "trainer": QLoRATrainer,
        },
        "full": {
            "name": "Full Fine-tuning",
            "description": "Train all parameters — best quality, most compute",
            "memory": "High (A100×8+)",
            "quality": "100% (baseline)",
            "presets": ["full-quick", "full-standard"],
            "trainer": None,  # Uses Phase 7 distributed trainer
        },
        "dpo": {
            "name": "DPO Alignment",
            "description": "Direct Preference Optimization — align to preferences",
            "memory": "Medium",
            "quality": "Comparable to RLHF",
            "presets": ["dpo-standard", "dpo-deep"],
            "trainer": DPOTrainer,
        },
        "distill": {
            "name": "Knowledge Distillation",
            "description": "Train small model from large teacher",
            "memory": "Medium-High (needs teacher)",
            "quality": "80-95% of teacher",
            "presets": ["distill-soft", "distill-hard"],
            "trainer": DistillationTrainer,
        },
        "instruct": {
            "name": "Instruction Tuning",
            "description": "Supervised fine-tuning on instructions",
            "memory": "Medium",
            "quality": "Excellent for chat/assistant use",
            "presets": ["instruct-fast", "instruct-standard"],
            "trainer": InstructionTuner,
        },
        "domain": {
            "name": "Domain Continued Pre-training",
            "description": "Extend base model on domain-specific data",
            "memory": "High",
            "quality": "Significant domain improvement",
            "presets": ["domain-code", "domain-japanese"],
            "trainer": None,
        },
        "curriculum": {
            "name": "Curriculum Learning",
            "description": "Progressive difficulty — easy → hard",
            "memory": "Medium",
            "quality": "Better convergence on complex tasks",
            "presets": ["curriculum-easy", "curriculum-full"],
            "trainer": None,
        },
        "multitask": {
            "name": "Multi-task Learning",
            "description": "Joint training on multiple objectives",
            "memory": "High",
            "quality": "Better generalization",
            "presets": ["multitask-standard"],
            "trainer": None,
        },
    }

    @classmethod
    def list_methods(cls) -> list[dict]:
        """List all training methods."""
        return [
            {"id": mid, "name": m["name"], "description": m["description"],
             "memory": m["memory"], "quality": m["quality"]}
            for mid, m in cls.METHODS.items()
        ]

    @classmethod
    def list_presets(cls) -> list[dict]:
        """List all presets."""
        return [
            {"id": pid, "method": p.method, "description": p.description,
             "lr": p.learning_rate, "batch_size": p.batch_size, "epochs": p.epochs}
            for pid, p in TRAINING_PRESETS.items()
        ]

    @classmethod
    def get_preset(cls, preset_id: str) -> Optional[TrainingPreset]:
        """Get a preset by ID."""
        return TRAINING_PRESETS.get(preset_id)

    @classmethod
    def recommend(cls, task: str, hardware: str = "auto") -> list[dict]:
        """Recommend training methods for a given task and hardware."""
        recommendations = []

        if hardware == "auto":
            try:
                import torch
                if torch.backends.mps.is_available():
                    hardware = "mps"
                elif torch.cuda.is_available():
                    hardware = "cuda"
                else:
                    hardware = "cpu"
            except Exception:
                hardware = "cpu"

        task_lower = task.lower()
        is_mps = hardware == "mps"

        if "chat" in task_lower or "instruct" in task_lower:
            if is_mps:
                recommendations.append({"method": "lora", "preset": "lora-standard",
                                        "reason": "Best fit for MPS: LoRA on instruction data"})
            else:
                recommendations.append({"method": "instruct", "preset": "instruct-standard",
                                        "reason": "Full instruction tuning"})

        if "code" in task_lower:
            recommendations.append({"method": "lora", "preset": "lora-deep",
                                    "reason": "Deep LoRA for code generation tasks"})
            if not is_mps:
                recommendations.append({"method": "domain", "preset": "domain-code",
                                        "reason": "Domain pre-training on code"})

        if "align" in task_lower or "preference" in task_lower:
            recommendations.append({"method": "dpo", "preset": "dpo-standard",
                                    "reason": "DPO alignment on preference data"})

        if "small" in task_lower or "distill" in task_lower:
            recommendations.append({"method": "distill", "preset": "distill-soft",
                                    "reason": "Distill from larger teacher model"})

        if "japanese" in task_lower or "日本語" in task_lower:
            recommendations.append({"method": "domain", "preset": "domain-japanese",
                                    "reason": "Domain adaptation for Japanese"})

        # Default: LoRA is always the safest bet
        if not recommendations:
            recommendations.append({"method": "lora", "preset": "lora-fast",
                                    "reason": "Quick LoRA — safe default for any task"})

        return recommendations
